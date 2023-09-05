
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import scipy.sparse as sp


from .model import Teacher_Features , Teacher_Edge , Student
from sklearn.cluster import KMeans , SpectralClustering , AgglomerativeClustering , DBSCAN , OPTICS , Birch
from .accuracy import get_accuracy
from .utils import students_t_kernel_euclidean , student_t_kernel_cosine , generate_targer_distribution
from .PPR_Matrix.ppr import topk_ppr_matrix
from .losses import  link_prediction_objectif
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from tqdm import tqdm

class Train : 
    def __init__(self , epochs ,  device  , num_nodes, input_dim , output_dim , hidden2 , nbr_clusters , beta_struct = 10 , beta_features = 10 , beta_student = 10 , teta = 0.5 ,dropout_rate=0.2 , lr=0.01 , weight_decay=0.0005 , clustering_method=KMeans):
        self.epochs = epochs
        self.device = device
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden2 = hidden2
        self.nbr_clusters = nbr_clusters
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
        self.teta = teta

        self.feature_teacher = Teacher_Features(self.num_nodes , self.input_dim , self.output_dim , self.hidden2 , self.dropout_rate , self.device)
        self.edge_teacher = Teacher_Edge(self.num_nodes , self.input_dim , self.output_dim , self.hidden2 , self.dropout_rate , self.device)
        self.student = Student(self.num_nodes , self.input_dim , self.output_dim , self.hidden2 ,self.hidden2 , self.hidden2 ,self.dropout_rate , self.device)

        self.feature_teacher.to(self.device)
        self.edge_teacher.to(self.device)
        self.student.to(self.device)

        self.feat_teach_optimizer = optim.Adam(self.feature_teacher.parameters() , lr=self.lr , weight_decay=self.weight_decay)
        self.feat_scheduler = ReduceLROnPlateau(self.feat_teach_optimizer , mode='min' , factor=0.5 , patience=5 , verbose=True)
        self.beta_features = beta_features

        self.edge_teach_optimizer = optim.Adam(self.edge_teacher.parameters() , lr=self.lr , weight_decay=self.weight_decay)
        self.edge_scheduler = ReduceLROnPlateau(self.edge_teach_optimizer , mode='min' , factor=0.5 , patience=5 , verbose=True)
        self.beta_struct = beta_struct

        self.student_optimizer = optim.Adam(self.student.parameters() , lr=self.lr , weight_decay=self.weight_decay) 
        self.student_scheduler = ReduceLROnPlateau(self.student_optimizer , mode='min' , factor=0.5 , patience=5 , verbose=True)
        self.beta_student = beta_student

        self.clustering_method = clustering_method

    """
    Pretrain the feeature teacher 
    """

    def pretrain_feat_teacher(self , feat_epochs , data ) :
        loadcheck = input("Do you want to load a checkpoint ? (y/n) : ")
        if loadcheck == 'y' :
            self.load_checkpoint(model='feat_teacher')
            print("Feature teacher checkpoint loaded")
        else :
            print("Training from scratch")

        print("making the clustering layer")

        output = []
        for batch in tqdm(data  , total=len(data)): 
            batch.to(self.device)
            X = batch.x.squeeze(1)
            pe_feat = batch.pe_feat
            with torch.no_grad() : 
                feat_teacher_output , _ = self.feature_teacher(X , pe_feat)
                output.append(feat_teacher_output)
        all_outputs = torch.cat(output , dim=0)

        clustering = self.clustering_method(self.nbr_clusters , n_init="auto")
        cluster_ids = clustering.fit_predict(all_outputs.cpu().detach().numpy())
        self.feature_teacher.Cluster_Layer = torch.tensor(clustering.cluster_centers_).to(self.device)
        self.feature_teacher.Cluster_Layer.requires_grad = False
        print("Clustering layer made")
            
            # clear memory
        del X, pe_feat, feat_teacher_output , output , all_outputs 
        torch.cuda.empty_cache()

        lowest_loss = np.inf

        for epoch in range(1 , 1+feat_epochs):
            print("starting epoch {}".format(epoch))
            all_outputs = []
            epoch_loss=0
            for batch in tqdm(data , desc="Epoch : {}".format(epoch) , total=len(data)):
                batch.to(self.device)
                X = batch.x.squeeze(1)
                
                pe_feat = batch.pe_feat

                feat_teacher_output , _ = self.feature_teacher(X, pe_feat)

                Q = students_t_kernel_euclidean(feat_teacher_output , self.feature_teacher.Cluster_Layer)
                P = generate_targer_distribution(Q)
                kl_loss = nn.KLDivLoss()(torch.log(Q) ,P.detach())
        
                mse_loss = 0
                if self.output_dim == self.input_dim :
                    mse_loss = nn.MSELoss()(feat_teacher_output , X)
                    with open("mse_loss.txt" , "a") as f :
                        f.write("Epoch : {} , Loss : {} \n".format(epoch , mse_loss.item()))
                        f.write("shape of the output : {} \n".format(feat_teacher_output.shape))
                        f.write("shape of the input : {} \n".format(X.shape))
                        f.write("shape of the pe_feat : {} \n".format(pe_feat.shape))
                        f.write("is their nan in the output : {} \n".format(torch.isnan(feat_teacher_output).any()))
                        f.write("is their nan in the input : {} \n".format(torch.isnan(X).any()))
                        

                loss = self.beta_features*kl_loss + mse_loss

                self.feat_teach_optimizer.zero_grad()
                loss.backward()
                self.feat_teach_optimizer.step()
                epoch_loss += loss.item()

                feat_teacher_output = feat_teacher_output.squeeze(0).detach().cpu()
                all_outputs.append(feat_teacher_output)


                # clear memory
                del X, pe_feat, feat_teacher_output, Q, P, kl_loss, mse_loss, loss
                torch.cuda.empty_cache()
            self.feat_scheduler.step(epoch_loss)
            if epoch_loss < lowest_loss :
                lowest_loss = epoch_loss
                self.save_checkpoint(epoch , model='best_feat_teacher')
                print("Best feature teacher checkpoint saved at epoch {}".format(epoch))
            if epoch_loss > lowest_loss :
                patience += 1
            else :
                patience = 0
            if patience > 10 :
                print("Early stopping")
                break

            all_outputs = torch.cat(all_outputs , dim=0)
            sil_score, db_score, ch_score = get_accuracy(all_outputs , self.nbr_clusters)

            print("Epoch : {} , Loss : {}".format(epoch , epoch_loss))
            print("Silhouette score : {} , Davies Bouldin score : {} , Calinski Harabasz score : {}".format(sil_score , db_score , ch_score))
            if epoch % 10 == 0 :
                self.save_checkpoint(epoch , model='feat_teacher')
                print("Feature teacher checkpoint saved at epoch {}".format(epoch))

            """
    pretrain the edge teacher 
    """

    def pretrain_edge_teacher(self , edge_epochs , data) :
        loadcheck = input("Do you want to load a checkpoint ? (y/n) : ")
        if loadcheck == 'y' :
            self.load_checkpoint(model='edge_teacher')
            print("edge teacher checkpoint loaded")
        else :
            print("Training from scratch")

        print("making the clustering layer")
        output = []
        for batch in tqdm(data  , total=len(data)) :
            batch.to(self.device)
            X = batch.x.squeeze(1)
            adj_aug = batch.adj_aug
            pe_feat = batch.pe_feat.to_dense()
            with torch.no_grad() : 
                edge_teacher_output , _ = self.edge_teacher( adj_aug, pe_feat , X)
                output.append(edge_teacher_output.squeeze(0))
            del adj_aug , pe_feat , edge_teacher_output
        all_outputs = torch.cat(output , dim=0)

        clustering = self.clustering_method(self.nbr_clusters , n_init="auto")
        cluster_ids = clustering.fit_predict(all_outputs.cpu().detach().numpy())
        self.edge_teacher.Cluster_Layer = torch.tensor(clustering.cluster_centers_).to(self.device)
        self.edge_teacher.Cluster_Layer.requires_grad = False
        print("Clustering layer made")
        print("the shape of the clustering layer is : {}".format(self.edge_teacher.Cluster_Layer.shape))
        del output , all_outputs
        torch.cuda.empty_cache()


        self.edge_teacher.train()

        lowest_loss = np.inf

        

        for epoch in range(1 , 1+edge_epochs):
            print("starting epoch {}".format(epoch))
            epoch_loss=0
            all_outputs = []
            for batch in tqdm(data , desc="Epoch : {}".format(epoch) , total=len(data)):
                batch.to(self.device)
                X = batch.x.squeeze(1)
                adj_aug = batch.adj_aug
                pe_feat = batch.pe_feat.to_dense()

                

                edge_teacher_output , _ = self.edge_teacher( adj_aug, pe_feat , X)
                edge_teacher_output = edge_teacher_output.squeeze(0)

                Q = students_t_kernel_euclidean(edge_teacher_output , self.edge_teacher.Cluster_Layer)
                P = generate_targer_distribution(Q)

                kl_loss = nn.KLDivLoss()(torch.log(Q) ,P.detach())

                mse_loss = 0
                if self.output_dim == self.input_dim :
                    mse_loss = nn.MSELoss()(edge_teacher_output , X)
                    # print the mse loss in another file
                    with open("mse_loss.txt" , "a") as f :
                        f.write("Epoch : {} , Loss : {} \n".format(epoch , mse_loss.item()))
                        f.write("shape of the output : {} \n".format(edge_teacher_output.shape))
                        f.write("shape of the input : {} \n".format(X.shape))
                        f.write("shape of the pe_feat : {} \n".format(pe_feat.shape))
                        f.write("is their nan in the output : {} \n".format(torch.isnan(edge_teacher_output).any()))
                        f.write("is their nan in the input : {} \n".format(torch.isnan(X).any()))


                loss = self.beta_struct*kl_loss + mse_loss

                self.edge_teach_optimizer.zero_grad()
                loss.backward()
                self.edge_teach_optimizer.step()

                epoch_loss += loss.item()
                edge_teacher_output = edge_teacher_output.squeeze(0).detach().cpu()
                all_outputs.append(edge_teacher_output)

                # clear memory
                del adj_aug , pe_feat , edge_teacher_output, Q, P, kl_loss, mse_loss, loss
                torch.cuda.empty_cache()
            self.edge_scheduler.step(epoch_loss)
            if epoch_loss < lowest_loss :
                lowest_loss = epoch_loss
                self.save_checkpoint(epoch , model='best_edge_teacher')
                print("Best edge teacher checkpoint saved at epoch {}".format(epoch))
            if epoch_loss > lowest_loss :
                patience += 1
            else :
                patience = 0
            if patience > 10 :
                print("Early stopping")
                break

            all_outputs = torch.cat(all_outputs , dim=0)
            sil_score, db_score, ch_score = get_accuracy(all_outputs , self.nbr_clusters)


            
            print("Epoch : {} , Loss : {}".format(epoch , epoch_loss))
            print("Silhouette score : {} , Davies Bouldin score : {} , Calinski Harabasz score : {}".format(sil_score , db_score , ch_score))
            if epoch % 10 == 0 :
                self.save_checkpoint(epoch , model='edge_teacher')
                print("Edge teacher checkpoint saved at epoch {}".format(epoch))


    """
    given a feature teacher and an edge/structure teacher train the student
    """

    def train_student(self , student_epochs , data ):
        loadcheck = input("Do you want to load a checkpoint ? (y/n) : ")
        if loadcheck == 'y' :
            self.load_checkpoint(model='student')
            print("student checkpoint loaded")
        else :
            print("Training from scratch")
        print("making the clustering layer for the student model")

        output = []
        for batch in tqdm(data, total=len(data)):
            batch.to(self.device)
            X = batch.x.squeeze(1)
            adj = batch.adj
            adj_aug = batch.adj_aug
            with torch.no_grad() :
                student_output, _ , _ , _= self.student(X , adj_aug)
                output.append(student_output.squeeze(0))
        all_outputs = torch.cat(output , dim=0)

        clustering = self.clustering_method(self.nbr_clusters , n_init="auto")
        cluster_ids = clustering.fit_predict(all_outputs.cpu().detach().numpy())
        self.student.Cluster_Layer = torch.tensor(clustering.cluster_centers_).to(self.device)
        self.student.Cluster_Layer.requires_grad = False
        print("Clustering layer for student model made")

        del output , all_outputs
        torch.cuda.empty_cache()


        self.student.train()

        lowest_loss = np.inf




        for epoch in range(1 , 1+student_epochs):
            print("starting epoch {}".format(epoch))
            epoch_loss=0
            epoch_clustering_loss = 0
            epoch_link_prediction_loss = 0

            for batch in tqdm(data , desc="Epoch : {}".format(epoch) , total=len(data)):
                batch.to(self.device)
                X = batch.x.squeeze(1)
                adj = batch.adj
                adj_aug = batch.adj_aug
                pe_feat = batch.pe_feat
                pe_feat_dense = pe_feat.to_dense()

                with torch.no_grad() :
                    _ , middle_representation_edge = self.edge_teacher(adj_aug ,pe_feat_dense , X)
                    _ , middle_representation_feat = self.feature_teacher(X , pe_feat)

                student_output , middle_representation , link_predictions , teta = self.student(X , adj_aug)
                student_output = student_output.squeeze(0)
                link_prediction = link_predictions.squeeze(0)
                
                
            
                Q = students_t_kernel_euclidean(student_output , self.student.Cluster_Layer)
                P = generate_targer_distribution(Q)

                link_prediction_loss = link_prediction_objectif(link_prediction , batch.edge_index)

                kl_loss = nn.KLDivLoss()(torch.log(Q) ,P.detach())

                struct_loss , feat_loss = self.student.loss(middle_representation , middle_representation_feat , middle_representation_edge) 
                
                clustering_loss = kl_loss + teta*struct_loss + (1-teta)*feat_loss
                
                mse_loss = 0
                if self.output_dim == self.input_dim :
                    mse_loss = nn.MSELoss()(student_output , X)
                
                
                loss =  10*link_prediction_loss + teta*struct_loss + (1-teta)*feat_loss

                with open("Student_Training_with_link_pred.txt" , "a") as f :
                    f.write("======================================== \n")
                    f.write("Epoch : {} , Loss : {} \n".format(epoch , link_prediction_loss.item()))
                    f.write("shape of the output : {} \n".format(student_output.shape))
                    f.write("shape of the input : {} \n".format(X.shape))
                    f.write("shape of the pe_feat : {} \n".format(pe_feat.shape))
                    f.write("is their nan in the output : {} \n".format(torch.isnan(student_output).any()))
                    f.write("is their nan in the input : {} \n".format(torch.isnan(X).any()))
                    f.write("is their nan in the pe_feat : {} \n".format(torch.isnan(pe_feat).any()))
                    f.write("is their nan in the pe_feat_dense : {} \n".format(torch.isnan(pe_feat_dense).any()))
                    f.write("is their nan in the Q : {} \n".format(torch.isnan(Q).any()))
                    f.write("is their nan in the P : {} \n".format(torch.isnan(P).any()))
                    f.write("is their nan in the kl_loss : {} \n".format(torch.isnan(kl_loss).any()))
                    f.write("is their nan in the struct_loss : {} \n".format(torch.isnan(struct_loss).any()))
                    f.write("is their nan in the feat_loss : {} \n".format(torch.isnan(feat_loss).any()))
                    f.write("is their nan in the clustering_loss : {} \n".format(torch.isnan(clustering_loss).any()))
                    f.write("is their nan in the loss : {} \n".format(torch.isnan(loss).any()))
                    f.write("the losses are : linkpred {} , mse {} , clustering {} , struct {} , feat {} \n".format(link_prediction_loss.item() , mse_loss.item() , clustering_loss.item() , struct_loss.item() , feat_loss.item()))
                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()
                epoch_loss += loss.item()
                epoch_clustering_loss += clustering_loss.item()
                epoch_link_prediction_loss += link_prediction_loss.item()


                # clear memory
                del X, adj, adj_aug, student_output, Q, P, kl_loss, struct_loss, feat_loss, loss , middle_representation , middle_representation_edge , middle_representation_feat  , clustering_loss , pe_feat , pe_feat_dense , link_prediction_loss , link_prediction
                torch.cuda.empty_cache()
            self.student_scheduler.step(epoch_loss)
            if epoch_loss < lowest_loss :
                lowest_loss = epoch_loss
                self.save_checkpoint(epoch , model='best_student')
                print("Best student checkpoint saved at epoch {}".format(epoch))
            if epoch_loss > lowest_loss :
                patience += 1
            else :
                patience = 0
            if patience > 10 :
                print("Early stopping")
                break
            


            print("Epoch : {} , Loss : {} , clustering_loss : {} , link_pred_loss : {}".format(epoch , epoch_loss , epoch_clustering_loss , epoch_link_prediction_loss))
            if epoch % 10 == 0 :
                self.save_checkpoint(epoch , model='student')
                print("Student model checkpoint saved at epoch {}".format(epoch))


    """
    Save checkpoint
    """

    def save_checkpoint(self , epoch :int , path = "/T2_GNN/save_checkpoint/" , model = 'student' ):
        if not os.path.exists(path):
            os.makedirs(path)

        if model == 'student':
            torch.save(self.student.state_dict() , path+'_student.pth')
        elif model == 'feat_teacher':
            torch.save(self.feature_teacher.state_dict() , path+'_feat_teacher.pth')
        elif model == 'edge_teacher':
            torch.save(self.edge_teacher.state_dict() , path+'_edge_teacher.pth')
        elif model == 'best_student':
            torch.save(self.student.state_dict() , path+'_best_student.pth')
        elif model == 'best_feat_teacher':
            torch.save(self.feature_teacher.state_dict() , path+'_best_feat_teacher.pth')
        elif model == 'best_edge_teacher':
            torch.save(self.edge_teacher.state_dict() , path+'_best_edge_teacher.pth')
        else : 
            raise ValueError("Model must be student or feat_teacher or edge_teacher")
        
    """
    Load checkpoint
    """

    def load_checkpoint(self , path = "/T2_GNN/save_checkpoint/" , model = 'student'):
        if not os.path.exists(path):
            print("No checkpoint found")
            print("Training from scratch")
            return
        
        if model == 'student':
            self.student.load_state_dict(torch.load(path+'_best_student.pth'))
        elif model == 'feat_teacher':
            self.feature_teacher.load_state_dict(torch.load(path+'_best_feat_teacher.pth'))
        elif model == 'edge_teacher':
            self.edge_teacher.load_state_dict(torch.load(path+'_best_edge_teacher.pth'))
        else : 
            raise ValueError("Model must be student or feat_teacher or edge_teacher")
        

    