# Packages
import sys, os
import json
import importlib
import numpy as np
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Appending src dir. for module import
sys.path.append(os.path.dirname(os.getcwd()))

# Modules
import helper_functions as f
importlib.reload(f)


def get_responses(vartest_dir: str, test_number: int, instance: str = 'uni'):
    """
    Gets all responses from var test.

    Args:
        vartest_dir (str): Full directory to vartests.
        test_number (int): Vartest test number.
        instance (str): Type of instance.

    Returns:
        list: List of all responses
    """
    instance_types = f.get_instance_types(instance=instance)

    test_dir = os.path.join(vartest_dir, f'test_{test_number}/')

    responses = {w: [f.file_to_string(os.path.join(test_dir, f'responses/{w}/', file)) for file in os.listdir(os.path.join(test_dir, f'responses/{w}/'))] for w in instance_types}
    json_responses = {w: [json.loads(responses[w][i]) for i in range(len(responses[w]))] for w in instance_types}

    return json_responses


def name_similarity(responses: dict) -> dict:
    """
    Calculating category name similarities.

    Args:
        responses (dict): Dictionary of JSON responses.

    Returns:
        dict: Dictionary of category cosine similarities.
    """
    new_values = {key: [] for key in responses.keys()}
    for key, items in responses.items():
        for item in items:
            combined_cats = " ".join(cat['category_name'] for cat in item['categories'])
            new_values[key].append(combined_cats)
    
    vectorizer = TfidfVectorizer()
    similarities = {}
    for key, combined_texts in new_values.items():
        if len(combined_texts) < 2:
            similarities[key] = None
            continue

        tfidf_matrix = vectorizer.fit_transform(combined_texts)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        upper_triangle = np.triu_indices_from(sim_matrix, k=1)
        upper_triangle_values = sim_matrix[upper_triangle]
        
        similarities[key] = list(upper_triangle_values)
    
    return similarities


def definition_similarity(responses: dict) -> list:
    """
    Calculates category definition similarity by matching pairs of definitions with the maximum cosine similarity, with respect to the matching scheme.

    Args:
        responses (dict): Dictionary of JSON responses.

    Returns:
        list: List of definition similarities.
    """
    final_matches = {key: [] for key in responses.keys()}
    for key in responses.keys():
        definitions_with_ids = []
        for replication_id, replication in enumerate(responses[key]):
            for category in replication['categories']:
                definitions_with_ids.append((replication_id, category['definition']))
    
        definitions = [item[1] for item in definitions_with_ids]
        replication_ids = [item[0] for item in definitions_with_ids]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(definitions)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # EX: (sim_score, rep_id1, rep_id2)
        similarity_scores = []
        for i, j in combinations(range(len(definitions)), 2):
            if replication_ids[i] != replication_ids[j]:
                similarity_scores.append((similarity_matrix[i, j], i, j))
        
        similarity_scores.sort(reverse=True, key=lambda x: x[0])
        
        matched = set()
        for sim, i, j in similarity_scores:
            if i not in matched and j not in matched:
                matched.add(i)
                matched.add(j)
                final_matches[key].append(sim)
    
    return final_matches

    