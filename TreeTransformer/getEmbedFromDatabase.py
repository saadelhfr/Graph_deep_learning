import chromadb


def initialiseDb(path ):
    """
    path : path to the database
    """
    client = chromadb.PersistentClient(path)
    collection = client.get_collection('nodes')
    return collection


def getNotNoneMetadata(metadataList):
    """
    metadataList : list of metadata dictionaries
    """
    return [metadata for metadata in metadataList if metadata is not None]
def getAttrFromDb(Collection , querry_embedding ):
    """
    query_embedding : list of embeddings of the query node
    """
    nbr_neighbors = 3
    queryrespnse = Collection.query (
        querry_embedding,
        n_results = nbr_neighbors,
    )

    return_dicts = {} 
    list_metadata = getNotNoneMetadata(queryrespnse["metadatas"][0])[0]

    #list_metadata = queryrespnse["metadatas"][0][0]
    print(list_metadata)
    if list_metadata is None:   
        return_dicts["user_name"] = "unknown"
        return_dicts["birthlocation"] = "unknown"
        return_dicts["deathlocation"] = "unknown"
        return_dicts["birthdatedecade"] = "unknown"
        return_dicts["deathdatedecade"] = "unknown"
        return_dicts["firstname"] = "unknown"
        return_dicts["lastnameatbirth"] = "unknown"
        return_dicts["gender"] = "unknown"
    
    return_dicts["user_name"] = list_metadata["user_name"]
    return_dicts["birthlocation"] = list_metadata["birthlocation"]
    return_dicts["deathlocation"] = list_metadata["deathlocation"]
    return_dicts["birthdatedecade"] = list_metadata["birthdatedecade"]
    return_dicts["deathdatedecade"] = list_metadata["deathdatedecade"]
    return_dicts["firstname"] = list_metadata["firstname"]
    return_dicts["lastnameatbirth"] = list_metadata["lastnameatbirth"]
    return_dicts["gender"] = list_metadata["gender"]
    return return_dicts



