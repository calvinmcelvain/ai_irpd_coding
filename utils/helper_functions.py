# Packages
import os, sys
import re
import pandas as pd
import importlib
from pydantic import BaseModel
from itertools import product
from datetime import datetime
from markdown_pdf import MarkdownPdf, Section

# Appending src dir. for module import
sys.path.append(os.path.dirname(os.getcwd()))

# Modules
import schemas.output_structures as outstr
importlib.reload(outstr)


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

    
def get_max_test_number(directory: str, prefix: str) -> int:
    """
    Gets next test number.
    """
    test_numbers = [
        int(re.findall(r'\d+', name)[0])
        for name in os.listdir(directory)
        if name.startswith(prefix) and re.findall(r'\d+', name)
    ]
    return max(test_numbers, default=0)


def load_json(file_path: str, json_schema: BaseModel) -> dict:
    """
    Load a JSON file from a given path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json_schema.model_validate_json(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        raise ValueError(f"Failed to parse JSON: {file_path}")


def check_directories(paths: list[str]) -> bool:
    """
    Check if all given directories exist.
    """
    return all(os.path.isdir(path) for path in paths)


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


def get_instances(instance: str) -> list[str]:
    """
    Returns the list of instances.
    """
    if instance == 'uni_switch':
        return ['uni', 'switch']
    return [instance]


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
        base_test_num = get_max_test_number(instance_dir, 'test_')
        test_dirs = [
            os.path.join(instance_dir, f"test_{base_test_num + (i if is_new_test else 0)}")
            for _ in stage
            for i in range(1, ra_num * treatment_num + 1)
        ]
        return sorted(test_dirs)
    
    if test_type == 'subtest':
        subtest_dir = os.path.join(output_dir, '_subtests')
        os.makedirs(subtest_dir, exist_ok=True)
        subtest_num = get_max_test_number(subtest_dir, '')
        return [os.path.join(subtest_dir, str(subtest_num + (1 if is_new_test else 0)))]
    
    if test_type == 'vartest':
        var_test_dir = os.path.join(output_dir, 'var_tests')
        os.makedirs(var_test_dir, exist_ok=True)
        test_num = get_max_test_number(var_test_dir, 'test_')
        return [os.path.join(var_test_dir, f"test_{test_num}")]
    
    raise ValueError("Invalid test_type. Must be 'test', 'subtest', or 'vartest'.")


def get_system_prompt(instance: str, ra: str, treatment: str, stage: str, prompt_path: str, test_path: str) -> dict:
    """
    Gets and returns dictionary of system prompts.
    """
    instances = get_instances(instance)
    
    system_prompts = {i: {} for i in instances}
    if stage not in {'1c'}:
        for i in instances:
            instance_types = get_instance_types(instance=i)

            if stage in {'2', '3'}:
                output = json_to_output(instance=i, test_dir=test_path, stage=stage)[i]
                markdown_prompt = file_to_string(f"{prompt_path}/{i}/{ra}/stg_{stage}_{treatment}.md")
                for t in instance_types:
                    system_prompts[i][t] = f"{markdown_prompt}\n{output[t]}"
            
            if stage in {'0', '1', '1r'}:
                for t in instance_types:
                    if stage == '1':
                        system_prompts[i][t] = file_to_string(f"{prompt_path}/{i}/{ra}/stg_{stage}_{treatment}_{t}.md")
                    else:
                        system_prompts[i][t] = file_to_string(f"{prompt_path}/{i}/{ra}/stg_{stage}_{treatment}.md")
    else:
        system_prompts = {}
        system_prompts['1c_1'] = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_1c_{treatment}.md")
        system_prompts['1c_2'] = file_to_string(f"{prompt_path}/{instance}/{ra}/stg_1r_{treatment}.md")

    return system_prompts


def get_user_prompt(instance: str, ra: str, treatment: str, stage: str, main_dir: str, test_dir: str, max_instances: int | None) -> dict:
    """
    Gets and returns dictionary of user prompts.
    """
    instances = get_instances(instance)
    
    user_prompts = {i: {} for i in instances}
    if stage not in {'1c'}:
        for i in instances:
            instance_types = get_instance_types(instance=i)
            data_frames = {
                t: pd.read_csv(os.path.join(main_dir, f"data/test/{i}_{treatment}_{ra}_{t}.csv")).astype({'window_number': int})
                for t in instance_types
            }

            if stage in {'0', '1', '2'}:
                if stage != '1':
                    user_prompts[i] = {
                        t: df[:max_instances] if max_instances else df 
                        for t, df in data_frames.items()
                    }
                else:
                    user_prompts[i] = {
                        t: df.to_dict('records')
                        for t, df in data_frames.items()
                    }
            
            if stage in {'1r'}:
                user_prompts[i] = json_to_output(test_dir=test_dir, instance=instance, stage=stage)[i]
            
            if stage in {'3'}:
                user_prompts[i] = {t: [] for t in instance_types}
                for t in instance_types:
                    df = data_frames[t][:max_instances] if max_instances else data_frames[t]
                    response_dir = os.path.join(test_dir, f"raw/stage_2_{t}/responses")
                    for file in os.listdir(response_dir):
                        if file.endswith("response.txt"):
                            json_response = load_json(os.path.join(response_dir, file), outstr.Stage_2_Structure)
                            summary = df[df['window_number'] == int(json_response.window_number)].to_dict('records')[0]
                            summary['assigned_categories'] = json_response.assigned_categories
                            user_prompts[i][t].append(summary)                
    else:
        part_1_exists = os.path.isdir(os.path.join(test_dir, "raw/stage_1c/part_1"))
        if not part_1_exists:
            if instance == 'uni_switch':
                combined_df = pd.read_csv(os.path.join(main_dir, f"data/test/{instance}_{treatment}_{ra}.csv"))
            else:
                instance_types = get_instance_types(instance=instance)
                data_frames = {
                    t: pd.read_csv(os.path.join(main_dir, f"data/test/{instance}_{treatment}_{ra}_{t}.csv")).astype({'window_number': int})
                    for t in instance_types
                }
                combined_df = pd.concat([df.assign(instance_type=(1 if t == instance_types[0] else 0)) for t, df in data_frames.items()], ignore_index=True)
            user_prompts = {'1c_1': combined_df.to_dict('records')}
        else:
            user_prompts = json_to_output(test_dir=test_dir, instance=instance, stage=stage)

    return user_prompts


def json_to_output(test_dir: str, instance: str, stage: str, output_format: str = "prompt") -> dict[str, str] | None:
    """
    Converts JSON objects to strings or PDFs based on the output format.
    """
    # Compute essential paths and variables
    test_num = get_test_number(test_dir=test_dir)
    raw_dir = os.path.join(test_dir, 'raw')
    
    instances = get_instances(instance)
    
    output = {i: "" for i in instances}
    text = {i: "" for i in instances}
    stage_1_data_allias = {i: {} for i in instances}
    stage_1r_data_allias = {i: {} for i in instances}
    stage_1r_keep_allias = {i: {} for i in instances}
    for i in instances:
        instance_types = get_instance_types(instance=i)

        stage_1_dirs = [os.path.join(raw_dir, f"stage_1_{t}") for t in instance_types]
        stage_1r_dirs = [os.path.join(raw_dir, f"stage_1r_{t}") for t in instance_types]
        stage_1c_dir = os.path.join(raw_dir, "stage_1c")
        stage_1c_parts = [os.path.join(stage_1c_dir, f"part_{k}") for k in range(1, 3)]

        # Load stage data
        if check_directories(stage_1_dirs):
            stage_1_data = {
                t: load_json(os.path.join(dir, f"t{test_num}_stg_1_{t}_response.txt"), outstr.Stage_1_Structure) 
                for t, dir in zip(instance_types, stage_1_dirs)
            }
            stage_1_data_allias[i] = stage_1_data
        if check_directories(stage_1r_dirs):
            stage_1r_data = {
                t: load_json(os.path.join(dir, f"t{test_num}_stg_1r_{t}_response.txt"), outstr.Stage_1r_Structure) 
                for t, dir in zip(instance_types, stage_1r_dirs)
            }
            stage_1r_data_allias[i] = stage_1r_data
            stage_1r_keep_decision = {
                t: {
                    cat.category_name: cat.keep_decision 
                    for cat in stage_1r_data[t].final_categories
                    } 
                for t in instance_types
            }
            stage_1r_keep_allias[i] = stage_1r_keep_decision
        if check_directories([stage_1c_parts[0]]):
            stage_1c_1_data = load_json(os.path.join(stage_1c_parts[0], f"t{test_num}_stg_1c_1_response.txt"), outstr.Stage_1_Structure)
        if check_directories([stage_1c_parts[1]]):
            stage_1c_2_data = load_json(os.path.join(stage_1c_parts[1], f"t{test_num}_stg_1c_2_response.txt"), outstr.Stage_1r_Structure)
            stage_1c_keep_decision = {cat.category_name: cat.keep_decision for cat in stage_1c_2_data.final_categories}

        # Prepare output
        if output_format == "prompt":
            output[i] = {t: "" for t in instance_types}
        if output_format == "pdf":
            text[i] = f"# Stage {stage} Categories\n\n"

        if stage == '1':
            for t in instance_types:
                categories = stage_1_data[t].categories
                text[i] += _format_categories(categories, initial_text=f"## {t.capitalize()} Categories\n\n")
        
        if stage == '1r':
            for t in instance_types:
                categories = stage_1_data[t].categories
                if output_format == "prompt":
                    output[i][t] += _format_categories(categories)
                else:
                    text[i] += _format_categories(categories, stage_1r_keep_decision[t], initial_text=f"## Kept {t.capitalize()} Categories\n\n")
                    text[i] += f"## Removed {t.capitalize()} Categories\n\n"
                    for cat in stage_1r_data[t].final_categories:
                        if not stage_1r_keep_decision[t][cat.category_name]:
                            text[i] += f"### {cat.category_name}\n\n"
                            text[i] += f"**Reasoning**: {cat.reasoning}\n\n"
        
        if stage in {'2', '3'}:
            if check_directories(stage_1r_dirs):
                if check_directories(stage_1c_parts):
                    stage_1c_categories = stage_1c_1_data.categories
                    stage_1c_category_names = {cat.category_name for cat in stage_1c_categories}
                    for t in instance_types:
                        output[i][t] += _format_categories(stage_1c_categories, stage_1c_keep_decision)
                        stage_1_categories = stage_1_data[t].categories
                        valid_categories = []
                        for cat in stage_1_categories:
                            if stage_1r_keep_decision[t].get(cat.category_name, False) and cat.category_name not in stage_1c_category_names:
                                valid_categories.append(cat)
                        output[i][t] += _format_categories(valid_categories)
                else:
                    for t in instance_types:
                        stage_1_categories = stage_1_data[t].categories
                        output[i][t] += _format_categories(stage_1_categories, stage_1r_keep_decision[t])
    
    if stage == '1c':
        categories = stage_1c_1_data.categories
        category_names = {cat.category_name for cat in categories}
        if output_format == 'prompt':
            output = {}
            output = {'1c_2': _format_categories(categories)}
        else:
            text = ""
            text += _format_categories(categories, stage_1c_keep_decision, initial_text="## Unified Categories\n\n")
            for i in instances:
                instance_types = get_instance_types(instance=i)
                text += f"## *{i.capitalize()} Categories*\n\n"
                for t in instance_types:
                    stage_1_categories = stage_1_data_allias[i][t].categories
                    valid_categories = []
                    for cat in stage_1_categories:
                        if stage_1r_keep_allias[i][t].get(cat.category_name, False) and cat.category_name not in category_names:
                            valid_categories.append(cat)
                    text += _format_categories(valid_categories, initial_text=f"## {t.capitalize()} Categories\n\n")
    
    # Save to PDF if necessary
    if output_format == "pdf":
        for i in instances:
            pdf = MarkdownPdf(toc_level=1)
            if instance == 'uni_switch' and stage not in {'1c'}:
                pdf.add_section(Section(text[i], toc=False))
                pdf.save(os.path.join(test_dir, f"t{test_num}_stg_{stage}_{i}_categories.pdf"))
            else:
                section = text[i] if stage not in {'1c'} else text
                pdf.add_section(Section(section, toc=False))
                pdf.save(os.path.join(test_dir, f"t{test_num}_stg_{stage}_categories.pdf"))
        return None

    return output


def _format_categories(categories: list, valid_categories: dict | None = None, initial_text: str = "") -> str:
    """
    Format category text for prompts.
    """
    formatted_text = initial_text
    for category in categories:
        if valid_categories is None or valid_categories.get(category.category_name, False):
            formatted_text += f" ### {category.category_name} \n\n"
            formatted_text += f" **Definition**: {category.definition}\n\n"
            try:
                formatted_text += f" **Examples**:\n\n"
                for idx, example in enumerate(category.examples, start=1):
                    formatted_text += f" {idx}. Window number: {example.window_number}, Reasoning: {example.reasoning}\n\n"
            except KeyError:
                pass
    return formatted_text


def write_test(test_dir: str, stage: str, instance_type: str, system: str, user: str, response: dict, window_number: str = None) -> None:
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
    
    if stage in {'2', '3'}:
        stage_dir = os.path.join(test_dir, "raw", f"stage_{stage}_{instance_type}")
        response_dir = os.path.join(stage_dir, "responses")
        prompt_dir = os.path.join(stage_dir, "prompts")
        os.makedirs(response_dir, exist_ok=True)
        os.makedirs(prompt_dir, exist_ok=True)
        if window_number in range(1001, 1005) or window_number in range(2001, 2005):
            write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{instance_type}_sys_prmpt.txt'), str(system))
        write_file(os.path.join(prompt_dir, f't{test_num}_{window_number}_user_prmpt.txt'), str(user))
        write_file(os.path.join(response_dir, f't{test_num}_{window_number}_response.txt'), str(response))
    
    if stage in {'1c'}:
        part = 1 if instance_type == '1c_1' else 2
        stage_dir = os.path.join(test_dir, "raw", "stage_1c", f"part_{part}")
        os.makedirs(stage_dir, exist_ok=True)
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{part}_sys_prmpt.txt'), str(system))
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{part}_user_prmpt.txt'), str(user))
        write_file(os.path.join(stage_dir, f't{test_num}_stg_{stage}_{part}_response.txt'), response)
    


def write_test_info(meta: dict, test_dir: str, model_info: object, data_file: dict, stage: str) -> None:
    """
    Writes test info.
    """
    test_num = get_test_number(test_dir=test_dir)
    instance = data_file['instance']
    instances = get_instances(instance)
    
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
    first_instance = next(iter(meta))
    first_key = next(iter(meta[first_instance]))
    test_info_file += f' Test date/time: {datetime.fromtimestamp(meta[first_instance][first_key].created).strftime('%Y-%m-%d %H:%M:%S')} \n'
    test_info_file += f' Instance: {instance} \n'
    test_info_file += f' Summary: {data_file['ra']} \n'
    test_info_file += f' Treatment: {data_file['treatment']} \n'
    test_info_file += f' System fingerprint: {meta[first_instance][first_key].system_fingerprint}\n'

    # Loop through each window
    total = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
    if instance == 'uni_switch' and stage in {'1c'}:
        instances = ['uni_switch']
    for i in instances:
        for key, value in meta[i].items():
            if instance == 'uni_switch' and stage not in {'1c'}:
                test_info_file += f" {i.upper()} {key.upper()} PROMPT USAGE: \n"
            else:
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
    

def build_gpt_output(test_dir: str, main_dir: str, instance: str, ra: str, treatment: str, stage: str, max_instances: int = None) -> None:
    """
    Builds GPT classification output for stages 2 or 3.
    """
    test_num = get_test_number(test_dir=test_dir)
    instances = get_instances(instance)
    
    for i in instances:
        test_df = get_user_prompt(instance=i, ra=ra, treatment=treatment, stage='2', main_dir=main_dir, test_dir=test_dir, max_instances=max_instances)[i]
        
        if stage not in {'2', '3'}:
            raise ValueError(f"Can only build GPT output if stage in ['2', '3']. Got: {stage}")
        
        instance_types = get_instance_types(instance=i)
        response_dirs = {t: os.path.join(test_dir, f"raw/stage_{stage}_{t}/responses") for t in instance_types}
        
        response_list = {t: [] for t in instance_types}
        df_list = []
        for t in instance_types:
            response_dir = response_dirs[t]
            for file in os.listdir(response_dir):
                file_path = os.path.join(response_dir, file)
                
                response_data = {}
                if stage == '3':
                    json_data = load_json(file_path, outstr.Stage_3_Structure)
                    for k in json_data.category_ranking:
                        response_data['reasoning'] = json_data.reasoning
                        response_data[k.category_name] = k.rank
                else:
                    json_data = load_json(file_path, outstr.Stage_2_Structure)
                    for k in json_data.assigned_categories:
                        response_data['reasoning'] = json_data.reasoning
                        response_data[k] = 1
                
                response_data['window_number'] = int(json_data.window_number)
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
            
            remove_columns = common_columns + [distinct_columns[i]]
            df_remove_cols = df.columns.intersection(remove_columns)
            df_dropped = df.drop(columns=df_remove_cols)
            category_columns = df_dropped.columns.to_list()
            
            rename_dict = {col: f'{t}_{col}' for col in category_columns}
            df = df.rename(columns=rename_dict)
            df_list.append(df)
        
        final_df = pd.concat([df_list[0], df_list[1]], ignore_index=True, sort=False)
        final_df = final_df.drop(columns=final_df.columns.intersection(['summary_1', 'summary_2'] + [distinct_columns[i]]))
        final_df = final_df.fillna(0)

        # Merge with raw DataFrame and save
        raw_df = pd.read_csv(os.path.join(main_dir, f'data/raw/{i}_{treatment}_{ra}.csv'))
        output_df = pd.merge(raw_df, final_df, on='window_number')
        if instance == 'uni_switch':
            output_df.to_csv(os.path.join(test_dir, f"t{test_num}_stg_{stage}_{i}_final_output.csv"), index=False)
        else:
            output_df.to_csv(os.path.join(test_dir, f"t{test_num}_stg_{stage}_final_output.csv"), index=False)
    

def merge_raw_data(instance: str, ra: str, main_dir: str) -> None:
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


def test_dfs(instance: str, ra: str, main_dir: str) -> None:
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