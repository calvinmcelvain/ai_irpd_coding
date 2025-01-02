# Packages
import sys, os
import importlib
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Appending src dir. for module import
sys.path.append(os.path.dirname(os.getcwd()))

# Modules
import helper_functions as f
import schemas.output_structures as outstr
importlib.reload(f)
importlib.reload(outstr)


def get_responses(vartest_dir: str, test_number: int, instance: str = 'uni'):
    """
    Gets all responses from var test.
    """
    instance_types = f.get_instance_types(instance=instance)

    test_dir = os.path.join(vartest_dir, f'test_{test_number}/')
    stage = next(folder for folder in os.listdir(test_dir) if folder.startswith('stage'))
    stage_dir = os.path.join(test_dir, stage)
    
    stage_schemas = {
        'stage_1': outstr.Stage_1_Structure,
        'stage_1r': outstr.Stage_1r_Structure,
    }
    schema = stage_schemas[stage]
    
    if stage not in {'stage_2', 'stage_3'}:
        responses = {
            w: [f.load_json(os.path.join(stage_dir, w, file), schema) 
                for file in os.listdir(os.path.join(stage_dir, w))] 
            for w in instance_types
        }
    else:
        responses = [pd.read_csv(os.path.join(stage_dir, file)) for file in os.listdir(stage_dir) if file.endswith('.csv')]

    return responses


def name_similarity(test_responses: list) -> pd.DataFrame:
    """
    Calculating category name similarities.
    """
    dfs = []
    for test in range(len(test_responses)):
        responses = test_responses[test]
        new_values = {key: [] for key in responses.keys()}
        for key, items in responses.items():
            for item in items:
                try:
                    category_names = [category.category_name.replace("_", " ") for category in item.categories]
                except AttributeError:
                    category_names = [category.category_name.replace("_", " ") for category in item.refined_categories]
                new_values[key].append(" ".join(category_names))
        
        vectorizer = TfidfVectorizer()
        data = []
        for key, combined_texts in new_values.items():
            if len(combined_texts) < 2:
                continue

            tfidf_matrix = vectorizer.fit_transform(combined_texts)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            upper_triangle = np.triu_indices_from(sim_matrix, k=1)
            upper_triangle_values = sim_matrix[upper_triangle]
            
            for sim in upper_triangle_values:
                data.append({
                    'instance_type': key,
                    'similarity': sim,
                    'test': test
                })
        
        dfs.append(pd.DataFrame(data))
    
    return pd.concat(dfs, axis=0, ignore_index=True)
    

def definition_similarity(test_responses: list) -> pd.DataFrame:
    """
    Calculates category definition similarity by matching pairs of definitions with the maximum cosine similarity, with respect to the matching scheme.
    """
    dfs = []
    for test in range(len(test_responses)):
        responses = test_responses[test]
        data = []
        for key in responses.keys():
            definitions_with_ids = []
            for replication_id, replication in enumerate(responses[key]):
                try:
                    for category in replication.categories:
                        definitions_with_ids.append((replication_id, category.definition))
                except AttributeError:
                    for category in replication.refined_categories:
                        definitions_with_ids.append((replication_id, category.definition))
        
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
                    data.append({
                        'instance_type': key,
                        'similarity': sim,
                        'test': test
                    })
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, axis=0, ignore_index=True)


def keep_decision(responses: dict) -> pd.DataFrame:
    """
    Gets keep decision from Stage 1r.
    """
    keep_decisions = {key: [] for key in responses.keys()}
    for key in responses.keys():
        replications = responses[key]
        cat_keep_decisions = {}
        for replication in replications:
            final_cats = replication['final_categories']
            for v in range(len(final_cats)):
                cat_name = final_cats[v]['category_name']
                if cat_name not in cat_keep_decisions:
                    cat_keep_decisions[cat_name] = []
                cat_keep_decisions[cat_name].append(final_cats[v]['keep_decision'])
        keep_decisions[key] = cat_keep_decisions
    
    data = []
    for key, cat_decisions in keep_decisions.items():
        for cat_name, decisions in cat_decisions.items():
            for decision in decisions:
                data.append({
                    'instance_type': key,
                    'category': cat_name,
                    'keep_decision': int(decision)
                })
    
    return pd.DataFrame(data)


def categorizations(responses: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Gets categorizations of Stage 2 or Stage 3 and returns a DataFrame of mean consistency.
    """
    instance_types = ['ucoop', 'udef']
    category_names = [i for i in responses[0].columns if i.startswith(instance_types[0]) or i.startswith(instance_types[1])] + ['window_number']
    
    trimmed_responses = [response[response.columns.intersection(category_names)] for response in responses]
    merged_responses = pd.concat(trimmed_responses).groupby('window_number').var().reset_index().drop(['window_number'], axis=1)
    
    data = []
    for instance_type in instance_types:
        mean_consistency = 1 - merged_responses.filter(like=instance_type).mean(axis=1)
        for value in mean_consistency:
            data.append({
                'instance_type': instance_type,
                'mean_consistency': value
            })
    
    return pd.DataFrame(data)
