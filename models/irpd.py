# Packages
import os, sys
import importlib
from itertools import product
from dotenv import load_dotenv

# Load configs (GPT api key, etc.)
load_dotenv(os.path.join(os.path.dirname(os.getcwd()), "config/configs.env"))

# Appending src dir. for module import
sys.path.append(os.path.dirname(os.getcwd()))

# Modules
import models.gpt as gpt_mod
import utils.helper_functions as f
import schemas.output_structures as outstr
importlib.reload(f)
importlib.reload(gpt_mod)
importlib.reload(outstr)


model_configs = {
    'base': gpt_mod.GPTConfig(seed=1240034),
    'res1': gpt_mod.GPTConfig(temperature=1.0),
    'res2': gpt_mod.GPTConfig(),
    'res3': gpt_mod.GPTConfig(temperature=1.0, seed=1240034),
}


class IRPD:
    def __init__(self, dir_path: str, gpt_model: gpt_mod.GPTConfig = None):
        """
        Initializing instance for TESTING.
        
        Args:
            dir_path (str): Main directory path.
            gpt_model (GPTConfig object, optional): GPT model confgis. Defaults to base.
        """
        self.PATH = dir_path
        if gpt_model:
            self.gpt = gpt_mod.GPT.default(config=gpt_model)
        else:
            self.gpt = gpt_mod.GPT.default(config=model_configs['base'])
        self.OUTPATH = os.path.join(dir_path, 'output')
        self.PROMPTPATH = os.path.join(dir_path, 'prompts')
        for path in [self.PATH, self.OUTPATH, self.PROMPTPATH]:
            os.makedirs(path, exist_ok=True)
    
    def reset_dir_path(self, dir_path: str) -> None:
        """Reset the main directory path."""
        self.PATH = dir_path
    
    def create_test_dfs(self, instance: str, ra: str):
        """
        Make test dataframes from raw summary or experimental data for IRPD testing.

        Args:
            instance (str): The instance type for the summaries in raw data.
            ras (list[str] | str): The RA or RAs in raw data.
            treatments (list[str] | str): The treatment or treatments in raw data.
        """
        # Valid options
        valid_instances = ['uni', 'uniresp', 'switch', 'first']
        valid_ras = ['thi', 'eli', 'both', 'exp']
        
        # Validate arguments
        f._validate_arg([ra], valid_ras, "ras")
        f._validate_arg([instance], valid_instances, "instance")
        
        # Creating trimmed data
        f.merge_raw_data(instance=instance, ra=ra, main_dir=self.PATH)
        f.test_dfs(instance=instance, ra=ra, main_dir=self.PATH)
    
    def run_test(self, instance: str, ras=None, stages=None, treatments=None, test_type: str = 'test', **kwargs):
        """
        Run IRPD test(s).

        Args:
            instance (str): The instance type for the summaries used in test.
            ras (list[str] | str, optional): The RA or RAs who wrote the summaries. Defaults to ['eli', 'thi', 'both'] or ['exp'] if stages includes 0. Defaults to None.
            stages (list[str] | str, optional): A set or single stage to be run. Defaults to ['1', '1r', '1c', '2', '3']. Defaults to None.
            treatments (list[str] | str, optional): A set or single treatment to be run. Defaults to ['noise', 'no_noise', 'merged']. Defaults to None.
            test_type (str, optional): The type of test to be run. Defaults to 'test'.
            **kwargs:
                - max_instances (int): Maximum number of instances used in Stage 2 and/or 3.
        """
        # Default values
        max_instances = None
        stages = f._ensure_list(stages or ['1', '1r', '1c', '2', '3'])
        treatments = f._ensure_list(treatments or ['noise', 'no_noise', 'merged'])
        if not '0' in stages:
            ras = f._ensure_list(ras or ['eli', 'thi', 'both'])
        else:
            ras = ['exp']
            print(f"Note 'ras' defaulted to {ras} since 'stages' include '0'. \n")

        # Valid options
        valid_stages = ['0', '1', '1r', '1c', '2', '3']
        valid_instances = ['uni', 'uniresp', 'switch', 'first', 'uni_switch']
        valid_ras = ['thi', 'eli', 'both', 'exp']
        valid_treatments = ['noise', 'no_noise', 'merged']
        valid_types = ['test', 'subtest']
        valid_kwargs = ['max_instances']
        valid_structures = {
            '0': outstr.Stage_0_Structure,
            '1': outstr.Stage_1_Structure,
            '1r': outstr.Stage_1r_Structure,
            '1c': outstr.Stage_1c_Structure,
            '2': outstr.Stage_2_Structure,
            '3': outstr.Stage_3_Structure
        }
        
        # Validate arguments
        f._validate_arg(stages, valid_stages, "stages")
        f._validate_arg(ras, valid_ras, "ras")
        f._validate_arg(treatments, valid_treatments, "treatments")
        f._validate_arg([instance], valid_instances, "instance")
        f._validate_arg([test_type], valid_types, "test_type")

        if (test_type == 'subtest' and (len(ras) > 1 or len(treatments) > 1)):
            raise ValueError(
                f"Invalid number of 'ras' or 'treatments' for 'subtest'. Must be equal to 1. Got: {ras}, {treatments}"
            )
        if ((len(ras) > 1 or len(treatments) > 1) and not ('0' in stages or '1' in stages)):
            raise ValueError(
                f"If multiple ras or treatments specified, stages must contain '0' or '1'. Got: {ras}, {treatments}, and {stages}"
            )
        for key, value in kwargs.items():
            if key not in valid_kwargs:
                raise ValueError(f"Invalid argument: '{key}'. Allowed arguments are: {valid_kwargs}.")
            elif key == 'max_instances':
                max_instances = value
        
        # Sorting stages, ras, & treatments list
        stages = sorted(stages, key=valid_stages.index)
        ras = sorted(ras, key=valid_ras.index)
        treatments = sorted(treatments, key=valid_treatments.index)
        
        # Getting test directories
        test_dirs = f.get_test_directory(
            output_dir=self.OUTPATH,
            instance=instance,
            test_type=test_type,
            stage=stages,
            ra_num=len(ras),
            treatment_num = len(treatments)
        )

        # Compressing ras, stages, & treatments
        test_zip = list(product(ras, treatments, stages))
        
        for ra, treatment, stage in test_zip:
            print(f"Running stage {stage}, {ra}, {treatment}....")
            
            # Getting the correct directory
            test_dir = test_dirs[(treatments.index(treatment) + ras.index(ra) * len(treatments))]
            
            # Compressed test info
            test_info = dict(instance=instance, ra=ra, treatment=treatment, stage=stage)
            
            # Getting prompts
            system = f.get_system_prompt(**test_info, prompt_path=self.PROMPTPATH, test_path=test_dir)
            user = f.get_user_prompt(**test_info, main_dir=self.PATH, test_dir=test_dir, max_instances=max_instances)

            meta = {t: 0 for t in system.keys()}
            if stage in ['1', '1r', '1c']:
                for t in system.keys():
                    # Setting max tokens
                    self.gpt.config = gpt_mod.GPTConfig(max_tokens=2000)
                    
                    # GPT requests
                    response_dict = self.gpt.gpt_request(
                        sys = str(system[t]),
                        user = str(user[t]),
                        output_structure = valid_structures[stage]
                    )

                    # Writing request info
                    f.write_test(
                        test_dir=test_dir,
                        stage=stage,
                        instance_type=t,
                        system=str(system),
                        user=str(user),
                        response=response_dict['response']
                    )
                    meta[t] = response_dict['meta']
                    
                    print(f"  Completed {t} instance!" if stage != '1c' else "  Completed instance!")
                
                f.json_to_output(test_dir=test_dir, stage=stage, instance=instance, output_format='pdf')
            elif stage in ['0', '2', '3']:
                for t in system.keys():
                    # Setting max tokens
                    self.gpt.config = gpt_mod.GPTConfig(max_tokens=600)
                    
                    # Calculate interval for printing dots
                    loader_interval = len(user[t]) // 10 if len(user[t]) > 10 else 1
                    print(f"  Making {t} stage {stage} requests", end="")
                    
                    # Iterative GPT requests
                    system_prompt = system[t]
                    stage_structure = valid_structures[stage]
                    for i, row in enumerate(user[t].itertuples(index=False) if stage != '3' else user[t], start=1):
                        
                        # Dot loader
                        if i % loader_interval == 0 or i == len(user[t]):
                            sys.stdout.write(".")
                            sys.stdout.flush()
                        
                        # GPT requests
                        response_dict = self.gpt.gpt_request(
                            sys = str(system_prompt),
                            user = str(row),
                            output_structure = stage_structure
                        )
                        
                        # Writing request info
                        f.write_test(
                            test_dir=test_dir,
                            stage=stage,
                            instance_type=t,
                            system=str(system),
                            user=row,
                            window_number=row['window_number'] if isinstance(row, dict) else row.window_number,
                            response=response_dict['response']
                        )
                        if i == 1:
                            meta[t] = response_dict['meta']
                        else:
                            meta[t].usage.completion_tokens += response_dict['meta'].usage.completion_tokens
                            meta[t].usage.prompt_tokens += response_dict['meta'].usage.prompt_tokens
                            meta[t].usage.total_tokens += response_dict['meta'].usage.total_tokens
                        
                    print(f"\n  Completed {t} instance!")
                print("  Aggregating responses and making dataframe...")
                f.build_gpt_output(test_dir=test_dir, main_dir=self.PATH, **test_info, max_instances=max_instances)
                
            # Writing test information to the specified directory
            f.write_test_info(meta=meta, test_dir=test_dir, model_info=self.gpt.config, data_file=test_info, stage=stage)
            print(f"  Stage {stage} complete!")
        
        print(f"Test complete, check {self.OUTPATH} for output.")
    
    def run_vartest(self, instance: str, ra: str, stage: str, treatment: str, N: int, **kwargs):
        """
        Run replication IRPD test.

        Args:
            instance (str): The instance type for the summaries used in test.
            ra (str): The RA who wrote the summaries.
            stage (str): The stage to be run.
            treatment (str): The treatment to be run.
            N (int): The number of replications.
            **kwargs:
                - max_instances (int): Maximum number of instances used in Stage 2 and/or 3.
        """
        # Default values
        max_instances = None
        
        # Valid options
        valid_stages = ['0', '1', '1r', '1c', '2', '3']
        valid_instances = ['uni', 'uniresp', 'switch', 'first']
        valid_ras = ['thi', 'eli', 'both', 'exp']
        valid_treatments = ['noise', 'no_noise', 'merged']
        valid_kwargs = ['max_instances']
        valid_structures = {
            '0': outstr.Stage_0_Structure,
            '1': outstr.Stage_1_Structure,
            '1r': outstr.Stage_1r_Structure,
            '1c': outstr.Stage_1c_Structure,
            '2': outstr.Stage_2_Structure,
            '3': outstr.Stage_3_Structure
        }
        
        # Validate arguments
        f._validate_arg([stage], valid_stages, "stage")
        f._validate_arg([ra], valid_ras, "ra")
        f._validate_arg([treatment], valid_treatments, "treatment")
        f._validate_arg([instance], valid_instances, "instance")
        
        if N < 1:
            raise ValueError(f"Invalid argument: 'N'. Must be greater than 0.")
        elif N == 1:
            print("Note: 'N' is equal to 1, this is not a replication test.")

        for key, value in kwargs.items():
            if key not in valid_kwargs:
                raise ValueError(f"Invalid argument: '{key}'. Allowed arguments are: {valid_kwargs}.")
            elif key == 'max_instances':
                if stage in ['2', '3']:
                    max_instances = value
                else:
                    print(f"Note: 'max_instances' is not used in Stage {stage}.")
        
        # Getting test directory
        test_dir = f.get_test_directory(
            output_dir=self.OUTPATH,
            instance=instance,
            test_type='vartest',
            stage=[stage],
        )