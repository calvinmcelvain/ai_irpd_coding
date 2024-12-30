import os
import re
import pandas as pd
import json
from itertools import product
from datetime import datetime
from markdown_pdf import MarkdownPdf, Section


def _validate_arg(arg: list[str], valid_values: list[str], name: str) -> None:
    """
    Validate that the argument is within valid values.
    """
    if not all(item in valid_values for item in arg):
        raise ValueError(f"'{name}' must be in {valid_values}. Got: {arg}")
    

def _ensure_list(item) -> list:
    """
    Ensure the input is a list.
    """
    return item if isinstance(item, list) else [item]


def file_to_string(file_path: str) -> str:
    """
    Return file contents as string.
    """
    with open(file_path, 'r') as file:
        k = file.read()
    return str(k)


def write_file(file_path: str, file_write) -> None:
  '''
  Writes files to path
  '''
  with open(file_path, 'w') as file:
    file.write(file_write)


def get_test_number(test_dir: str) -> int:
    """
    Gets test number.
    """
    match = re.search(r'\d+', test_dir)
    if match:
        test_number = int(match.group())
    else:
        raise ValueError(f"No test number found in directory: {test_dir}")
    
    return test_number

    
def get_next_test_number(directory: str, prefix: str) -> int:
    """
    Gets next test number.
    """
    test_numbers = [
        int(re.findall(r'\d+', name)[0])
        for name in os.listdir(directory)
        if name.startswith(prefix) and re.findall(r'\d+', name)
    ]
    return max(test_numbers, default=0) + 1


def get_test_directory(output_dir: str, test_type: str, stage: list = None, instance: str = None, ra_num: int = 1, treatment_num: int = 1) -> list[str]:
    """
    Creates test directory path.
    """
    if test_type != 'vartest' and (instance is None or stage is None):
        raise ValueError("Instance and Stage must be provided for test types other than 'vartest'.")
    
    is_new_test = test_type != 'vartest' and any(s in {'0', '1'} for s in stage)
    
    if test_type == 'test':
        instance_dir = os.path.join(output_dir, instance)
        os.makedirs(instance_dir, exist_ok=True)
        base_test_num = get_next_test_number(instance_dir, 'test_')
        test_dirs = [
            os.path.join(instance_dir, f"test_{base_test_num + (i if is_new_test else 0)}")
            for _ in stage
            for i in range(1, ra_num * treatment_num + 1)
        ]
        return sorted(test_dirs)
    
    if test_type == 'subtest':
        subtest_dir = os.path.join(output_dir, '_subtests')
        os.makedirs(subtest_dir, exist_ok=True)
        subtest_num = get_next_test_number(subtest_dir, '')
        return [os.path.join(subtest_dir, str(subtest_num + (1 if is_new_test else 0)))]
    
    if test_type == 'vartest':
        var_test_dir = os.path.join(output_dir, 'var_tests')
        os.makedirs(var_test_dir, exist_ok=True)
        test_num = get_next_test_number(var_test_dir, 'test_')
        return [os.path.join(var_test_dir, f"test_{test_num}")]
    
    raise ValueError("Invalid test_type. Must be 'test', 'subtest', or 'vartest'.")


def get_instance_types(instance: str) -> list[str]:
    """
    Returns the list of instance types.
    """
    instance_types = []
    if instance in ['uni', 'uniresp']:
        instance_types = ['ucoop', 'udef']
    
    if instance in ['switch', 'first']:
        instance_types = ['coop', 'def']
    
    return instance_types


def get_system_prompt(instance: str, ra: str, treatment: str, stage: str, prompt_path: str, test_path: str) -> dict:
    """
    Gets and returns dictionary of system prompts.
    """
    instance_types = get_instance_types(instance=instance)
    system_prompts = {}

    if stage in {'2', '3'}:
        output = json_to_output(instance=instance, test_dir=test_path, stage=stage)
        for t in instance_types:
            markdown_prompt = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_{stage}_{treatment}.md")
            system_prompts[t] = f"{markdown_prompt}\n{output[t]}"
    
    if stage in {'0', '1', '1r'}:
        for t in instance_types:
            if stage == '1':
                system_prompts[t] = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_{stage}_{treatment}_{t}.md")
            else:
                system_prompts[t] = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_{stage}_{treatment}.md")
    
    if stage in {'1c'}:
        system_prompts['1c_1'] = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_1c_{treatment}.md")
        system_prompts['1c_2'] = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_1r_{treatment}.md")

    return system_prompts


def get_user_prompt(instance: str, ra: str, treatment: str, stage: str, main_dir: str, test_dir: str, max_instances: int | None) -> dict:
    """
    Gets and returns dictionary of user prompts.
    """
    instance_types = get_instance_types(instance=instance)
    data_frames = {
        t: pd.read_csv(os.path.join(main_dir, f"data/test/{instance}_{treatment}_{ra}_{t}.csv")).astype({'window_number': int})
        for t in instance_types
    }

    if stage in {'0', '1', '2'}:
        user_prompts = {t: df[:max_instances].to_dict('records') if max_instances else df.to_dict('records') for t, df in data_frames.items()}
        if stage != '1':
            user_prompts = {t: df[:max_instances] if max_instances else df for t, df in data_frames.items()}
    
    if stage in {'1c'}:
        part_1_exists = os.path.isdir(os.path.join(test_dir, "raw/stage_1c/part_1"))
        if not part_1_exists:
            combined_df = pd.concat([df.assign(instance_type=(1 if t == instance_types[0] else 0)) for t, df in data_frames.items()], ignore_index=True)
            user_prompts = {'1c_1': combined_df.to_dict('records')}
        else:
            user_prompts = {'1c_2': json_to_output(test_dir=test_dir, instance=instance, stage=stage)}
    
    if stage in {'1r'}:
        user_prompts = json_to_output(test_dir=test_dir, instance=instance, stage=stage)
    
    if stage in {'3'}:
        user_prompts = {t: [] for t in instance_types}
        for t in instance_types:
            df = data_frames[t][:max_instances] if max_instances else data_frames[t]
            response_dir = os.path.join(test_dir, f"raw/stage_2_{t}/responses")
            for file in os.listdir(response_dir):
                if file.endswith("response.txt"):
                    json_response = json.loads(file_to_string(os.path.join(response_dir, file)))
                    summary = df[df['window_number'] == int(json_response['window_number'])].to_dict('records')[0]
                    summary['assigned_categories'] = json_response['assigned_categories']
                    user_prompts[t].append(summary)

    return user_prompts


def load_json(file_path: str) -> dict:
    """
    Load a JSON file from a given path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON: {file_path}")


def check_directories(*paths) -> bool:
    """
    Check if all given directories exist.
    """
    return all(os.path.isdir(path) for path in paths)


def json_to_output(test_dir: str, instance: str, stage: str, output_format: str = "prompt"):
    """
    Converts JSON objects to strings or PDFs based on the output format.
    """
    # Getting test number, instance types, and raw directory
    test_num = get_test_number(test_dir=test_dir)
    instance_types = get_instance_types(instance=instance)
    raw_dir = os.path.join(test_dir, 'raw')
    
    # Stage existince
    stage_1_exists = os.path.isdir(os.path.join(raw_dir, f"stage_1_{instance_types[0]}"))
    stage_1r_exists = os.path.isdir(os.path.join(raw_dir, f"stage_1r_{instance_types[0]}"))
    stage_1c_1_exists = os.path.isdir(os.path.join(raw_dir, "stage_1c", "part_1"))
    stage_1c_2_exists = os.path.isdir(os.path.join(raw_dir, "stage_1c", "part_2"))
    if not (stage_1r_exists or stage_1_exists):
        raise FileExistsError(f"Stage1, Stage 1r, and/or Stage 1c does not exist in {test_dir}")
    
    # Load stage 1 responses
    stage_1 = {
        t: json.loads(file_to_string(os.path.join(raw_dir, f"stage_1_{t}/t{test_num}_stg_1_{t}_response.txt")))
        for t in instance_types
    }
    
    # Initialize output variables
    if output_format == "prompt":
        output = {t: "" for t in instance_types}
    else:
        pdf = MarkdownPdf(toc_level=1)
        text = f"# Stage {stage} Categories\n\n"
        initial = {t: '' for t in instance_types}
    
    # Handle different stages
    if (stage == '1' and output_format != 'prompt') or (stage == '1r' and output_format == 'prompt') or (stage in {'2', '3'} and output_format == 'prompt' and not (stage_1r_exists)):
        for t in instance_types:
            categories = stage_1[t]['categories']
            formatted = _format_categories(categories, initial_text=f"\n\n ## {t.capitalize()} Categories \n\n")
            if output_format == "prompt":
                output[t] = formatted
            else:
                initial[t] = formatted
        if output_format == "pdf":
            text += ''.join(initial.values())
    elif (stage == '1c' and output_format == 'prompt' and not stage_1c_2_exists):
        stage_1c_1 = json.loads(file_to_string(os.path.join(raw_dir, f"stage_1c/part_1/t{test_num}_stg_1c_1_response.txt")))
        categories = stage_1c_1['categories']
        formatted = _format_categories(categories, initial_text=f"\n\n ## Categories \n\n")
        output = formatted
    elif (stage == '1c' and output_format != 'prompt' and (stage_1c_2_exists and stage_1r_exists)) or (stage in {'2', '3'} and output_format == 'prompt' and stage_1c_2_exists):
        stage_1c_1 = json.loads(file_to_string(os.path.join(raw_dir, f"stage_1c/part_1/t{test_num}_stg_1c_1_response.txt")))
        stage_1c_2 = json.loads(file_to_string(os.path.join(raw_dir, f"stage_1c/part_2/t{test_num}_stg_1c_2_response.txt")))
        initial_categories = stage_1c_1['categories']
        valid_unified_categories = {cat['category_name']: cat['keep_decision'] for cat in stage_1c_2['final_categories']}
        
        initial_text = "\n\n ## Unified Categories \n\n" if output_format != "prompt" else ''
        unified_categoried_formatted = _format_categories(initial_categories, valid_unified_categories, initial_text=initial_text)
        
        stage_1r = {
        t: json.loads(file_to_string(os.path.join(raw_dir, f"stage_1r_{t}/t{test_num}_stg_1r_{t}_response.txt")))
        for t in instance_types
        }
        for t in instance_types:
            final_categories = []
            valid_categories = {cat['category_name']: cat['keep_decision'] for cat in stage_1r[t]['final_categories']}
            for category in stage_1[t]['categories']:
                if valid_categories.get(category['category_name'], False) and category['category_name'] not in valid_unified_categories:
                    final_categories.append(category)
            if output_format == "prompt":
                unique_text = _format_categories(final_categories)
                output[t] = unique_text + unified_categoried_formatted
            else:
                unique_text = _format_categories(final_categories, initial_text=f"\n\n ## Unique {t.capitalize()} Categories \n\n")
                initial[t] = unique_text
        
        if output_format == "pdf":
            text += unified_categoried_formatted + ''.join(initial.values())
        
    elif (stage == '1r' and output_format != 'prompt') or (stage in {'2', '3'} and output_format == 'prompt' and not stage_1c_2_exists):
        stage_1r = {
            t: json.loads(file_to_string(os.path.join(raw_dir, f"stage_1r_{t}/t{test_num}_stg_1r_{t}_response.txt")))
            for t in instance_types
        }
        for t in instance_types:
            categories = stage_1[t]['categories']
            valid_categories = {cat['category_name']: cat['keep_decision'] for cat in stage_1r[t]['final_categories']}
            formatted = _format_categories(categories, valid_categories, f"\n\n ## {t.capitalize()} Categories \n\n")
            if output_format == "prompt":
                output[t] = formatted
            else:
                initial[t] = formatted + f"\n\n ## Removed {t.capitalize()} Categories \n\n"
                for cat in stage_1r[t]['final_categories']:
                    if not cat['keep_decision']:
                        initial[t] += f" ### {cat['category_name']} \n\n"
                        initial[t] += f" **Reasoning**: {cat['reasoning']}\n\n"
        if output_format == "pdf":
            text += ''.join(initial.values())
        else:
            output = {t: ''.join(output[t]) for t in instance_types}
    
    # Save PDF if required
    if output_format == "pdf":
        pdf.add_section(Section(text, toc=False))
        pdf.save(os.path.join(test_dir, f"t{test_num}_stg_{stage}_categories.pdf"))
        return None  # PDFs don't return text
    
    return output


def _format_categories(categories: list, valid_categories: dict | None = None, initial_text: str = "") -> str:
    """
    Format category text for prompts.
    """
    formatted_text = initial_text
    for category in categories:
        if valid_categories is None or valid_categories.get(category['category_name'], False):
            formatted_text += f" ### {category['category_name']} \n\n"
            formatted_text += f" **Definition**: {category['definition']}\n\n"
            try:
                formatted_text += f" **Examples**:\n\n"
                for idx, example in enumerate(category['examples'], start=1):
                    formatted_text += f" {idx}. Window number: {example['window_number']}, Reasoning: {example['reasoning']}\n\n"
            except KeyError:
                pass
    return formatted_text


def write_test(test_dir: str, stage: str, instance_type: str, system: dict, user: dict, response: dict, window_number: str = None):
    """
    Writes the raw prompts & outputs for tests.
    """
    test_num = get_test_number(test_dir=test_dir)
    
    # Raw response & prompts
    if stage in {'1', '1r'}:
        stage_dir = os.path.join(test_dir, "raw", f"stage_{stage}_{instance_type}")
        os.makedirs(stage_dir, exist_ok=True)
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{instance_type}_sys_prmpt.txt'), str(system))
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{instance_type}_user_prmpt.txt'), str(user))
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{instance_type}_response.txt'), str(response))
    elif stage in {'2', '3'}:
        stage_dir = os.path.join(test_dir, "raw", f"stage_{stage}_{instance_type}")
        response_dir = os.path.join(stage_dir, "responses")
        prompt_dir = os.path.join(stage_dir, "prompts")
        os.makedirs(response_dir, exist_ok=True)
        os.makedirs(prompt_dir, exist_ok=True)
        if window_number in range(1001, 1005) or window_number in range(2001, 2005):
            write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{instance_type}_sys_prmpt.txt'), str(system))
        write_file(os.path.join(prompt_dir, f't{test_num}_{window_number}_user_prmpt.txt'), str(user))
        write_file(os.path.join(response_dir, f't{test_num}_{window_number}_response.txt'), str(response))
    elif stage == '1c':
        part = 1 if instance_type == '1c_1' else 2
        stage_dir = os.path.join(test_dir, "raw", "stage_1c", f"part_{part}")
        os.makedirs(stage_dir, exist_ok=True)
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{part}_sys_prmpt.txt'), str(system))
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{part}_user_prmpt.txt'), str(user))
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{part}_response.txt'), response)
    


def write_test_info(meta: dict, test_dir: str, model_info: object, data_file: dict, stage: str):
    """
    Writes test info.
    """
    test_num = get_test_number(test_dir=test_dir)
    
    test_info_file = "MODEL INFORMATION: \n\n"
    test_info_file += f" Model: {model_info.model} \n"
    test_info_file += f" Termperature: {model_info.temperature} \n"
    test_info_file += f" Max-tokens: {model_info.max_tokens} \n"
    test_info_file += f" Seed: {model_info.seed} \n"
    test_info_file += f" Top-p: {model_info.top_p} \n"
    test_info_file += f" Frequency penalty: {model_info.frequency_penalty} \n"
    test_info_file += f" Presence penalty: {model_info.presence_penalty} \n\n\n"
    
    # Initialize a string to store formatted test info content
    test_info_file += "TEST INFORMATION:\n\n"
    
    # Getting test time
    first_key = next(iter(meta))
    test_info_file += f' Test date/time: {datetime.fromtimestamp(meta[first_key].created).strftime('%Y-%m-%d %H:%M:%S')} \n'
    test_info_file += f' Instance: {data_file['instance']} \n'
    test_info_file += f' Summary: {data_file['ra']} \n'
    test_info_file += f' Treatment: {data_file['treatment']} \n'
    test_info_file += f' System fingerprint: {meta[first_key].system_fingerprint}\n'

    # Loop through each window
    total = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
    for key, value in meta.items():
        test_info_file += f" {key.upper()} PROMPT USAGE: \n"
        test_info_file += f"   Completion tokens: {value.usage.completion_tokens} \n"
        test_info_file += f"   Prompt tokens: {value.usage.prompt_tokens} \n"
        test_info_file += f"   Total tokens: {value.usage.total_tokens} \n"
        total['completion_tokens'] += value.usage.completion_tokens
        total['prompt_tokens'] += value.usage.prompt_tokens
        total['total_tokens'] += value.usage.total_tokens
    
    # Writing totals
    test_info_file += f" TOTAL PROMPT USAGE: \n"
    test_info_file += f"   Completion tokens: {total['completion_tokens']} \n"
    test_info_file += f"   Prompt tokens: {total['prompt_tokens']} \n"
    test_info_file += f"   Total tokens: {total['total_tokens']} \n"

    # Define the directory and file path for the test info file
    info_dir = os.path.join(test_dir, f'raw/t{test_num}_stg{stage}_test_info.txt')

    # Write the formatted test info to the file
    write_file(file_path=info_dir, file_write=test_info_file)
    

def build_gpt_output(test_dir: str, main_dir: str, instance: str, ra: str, treatment: str, stage: str, max_instances: int = None):
    test_num = get_test_number(test_dir=test_dir)
    test_df = get_user_prompt(instance=instance, ra=ra, treatment=treatment, stage='2', main_dir=main_dir, test_dir=test_dir, max_instances=max_instances)
    
    if stage not in {'2', '3'}:
        raise ValueError(f"Can only build GPT output if stage in ['2', '3']. Got: {stage}")
    
    instance_types = get_instance_types(instance=instance)
    response_dirs = {t: os.path.join(test_dir, f"raw/stage_{stage}_{t}/responses") for t in instance_types}
    
    response_list = {t: [] for t in instance_types}
    df_list = []
    for t in instance_types:
        response_dir = response_dirs[t]
        for file in os.listdir(response_dir):
            file_path = os.path.join(response_dir, file)
            response = file_to_string(file_path=file_path)
            response_data = json.loads(response)
            
            if stage == '3':
                for i in response_data['category_ranking']:
                    response_data[i['category_name']] = i['rank']
                del response_data['category_ranking']
            else:
                for i in response_data['assigned_categories']:
                    response_data[i] = 1
                del response_data['assigned_categories']
            
            response_data['window_number'] = int(response_data['window_number'])
            response_list[t].append(response_data)
        
        df = pd.DataFrame.from_records(response_list[t])
        df = pd.merge(test_df[t], df, on='window_number', how='outer')
        
        common_columns = ['summary_1', 'summary_2', 'window_number', 'reasoning']
        distinct_columns = {
            'first': 'cooperation',
            'switch': 'cooperation',
            'uniresp': 'unilateral_other_cooperation',
            'uni': 'unilateral_cooperation'
        }
        
        remove_columns = common_columns + [distinct_columns[instance]]
        df_remove_cols = df.columns.intersection(remove_columns)
        df_dropped = df.drop(columns=df_remove_cols)
        category_columns = df_dropped.columns.to_list()
        
        rename_dict = {col: f'{t}_{col}' for col in category_columns}
        df = df.rename(columns=rename_dict)
        df_list.append(df)
    
    final_df = pd.concat([df_list[0], df_list[1]], ignore_index=True, sort=False)
    final_df = final_df.drop(columns=final_df.columns.intersection(['summary_1', 'summary_2'] + [distinct_columns[instance]]))
    final_df = final_df.fillna(0)

    # Merge with raw DataFrame and save
    raw_df = pd.read_csv(os.path.join(main_dir, f'data/raw/{instance}_{treatment}_{ra}.csv'))
    output_df = pd.merge(raw_df, final_df, on='window_number')
    output_df.to_csv(os.path.join(test_dir, f"t{test_num}_stg_{stage}_final_output.csv"), index=False)
    

def merge_raw_data(instance: str, ra: str, main_dir: str):
    '''
    Function that merges the treatments for summary data. Saves to raw folder.
    '''
    treatments = ['noise', 'no_noise']
    ras = ['thi', 'eli']
    
    if ra != 'both':
        dfs = {}
        for treatment in treatments:
            file_path = os.path.join(main_dir, f'data/raw/{instance}_{treatment}_{ra}.csv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            dfs[treatment] = pd.read_csv(file_path)
    else:
        all_dfs = {}
        for r, treatment in list(product(ras, treatments)):
            file_path = os.path.join(main_dir, f'data/raw/{instance}_{treatment}_{r}.csv')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            all_dfs[f"{treatment}_{r}"] = pd.read_csv(file_path)
        
        dfs = {}
        for treatment in treatments:
            dfs[treatment] = pd.merge(all_dfs[f"{treatment}_eli"], all_dfs[f"{treatment}_thi"], how='outer')
        
        for treatment in treatments:
            dfs[treatment].to_csv(os.path.join(main_dir,f'data/raw/{instance}_{treatment}_both.csv'), index=False)
        
    noise = dfs['noise']
    no_noise = dfs['no_noise']
    merged_df = pd.concat([no_noise, noise], ignore_index=True, sort=False)
    merged_df.to_csv(os.path.join(main_dir, f'data/raw/{instance}_merged_{ra}.csv'), index=False)


def test_dfs(instance: str, ra: str, main_dir: str):
    """
    Function that creates test data. Saves to test folder.
    """
    noise = pd.read_csv(os.path.join(main_dir, f'data/raw/{instance}_noise_{ra}.csv'))
    no_noise = pd.read_csv(os.path.join(main_dir,f'data/raw/{instance}_no_noise_{ra}.csv'))
    merged = pd.read_csv(os.path.join(main_dir,f'data/raw/{instance}_merged_{ra}.csv'))
    df_set = {
        'noise': noise,
        'no_noise': no_noise,
        'merged': merged
    }
    
    keep_columns = {
        1: ['summary_1', 'summary_2', 'window_number', 'cooperate', 'treatment'],
        2: ['summary_1', 'summary_2', 'window_number', 'unilateral_other_cooperate', 'unilateral_other_defect', 'treatment'],
        3: ['summary_1', 'summary_2', 'window_number', 'unilateral_cooperate', 'unilateral_defect', 'treatment']
    }
    summary_columns = ['summary_1', 'summary_2']
    window_columns = {
        1: {'coop': 'cooperate', 'def': 'defect'},
        2: {'ucoop': 'unilateral_other_cooperate', 'udef': 'unilateral_other_defect'},
        3: {'ucoop': 'unilateral_cooperate', 'udef': 'unilateral_defect'}
    }
    column_map = {
        'first': 1,
        'switch': 1,
        'uniresp': 2,
        'uni': 3
    }
        
    
    keep_columns = keep_columns[column_map[instance]]
    window_columns = window_columns[column_map[instance]]
    
    instance_types = get_instance_types(instance=instance)
    
    for treatment, df in df_set.items():
        df = df[df.columns.intersection(keep_columns)]
        if instance in {'switch', 'first'}:
            df = df.copy()
            df.loc[:, 'defect'] = df['cooperate'].apply(lambda x: 1 if x == 0 else 0)
        
        df_column = df.columns.intersection(summary_columns)
        df.loc[:, df_column] = df[df_column].replace(',', '', regex=True)
        
        for instance_type in instance_types:
            instance_df = df.loc[(df[window_columns[instance_type]] == 1)].drop(window_columns.values(), axis=1)
            instance_df.to_csv(os.path.join(main_dir, f'data/test/{instance}_{treatment}_{ra}_{instance_type}.csv'), index=False)