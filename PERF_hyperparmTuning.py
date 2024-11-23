from CLEAN_functions import format_timestamp
from collections import OrderedDict
import datetime as dt
import numpy as np
import pandas as pd
import pm4py as pm
import sys
import multiprocessing as mp
import itertools
from itertools import repeat
import copy

timestmp = dt.datetime.now().strftime("%y%m%d-%H-%M")


def theta_set (theta_min = float, theta_max = float, step_size = float) -> list:

        # Ensure theta_min is a valid float and within the range [0, 1]
        try:
                theta_min = float(theta_min)
        except (ValueError, TypeError):
                theta_min = 0  # If conversion fails, default to 0

        if theta_min < 0 or theta_min > 1:
                theta_min = 0

        # Ensure theta_max is a valid float and within the range [0, 1]
        try:
                theta_max = float(theta_max)
        except (ValueError, TypeError):
                theta_max = 1  # If conversion fails, default to 1

        if theta_max < 0 or theta_max > 1:
                theta_max = 1

        # Test if step_size is a valid float and within the range [0, 1],
        # else, abort the program
        try:
                step_size = float(step_size)
                if step_size < 0 or step_size > 1:
                        raise ValueError
        except (ValueError, TypeError):
                print('ERROR: STEP-SIZE:\nPlease indicate a step-size between 0 and 1')
                sys.exit()

        theta_set = np.round(np.arange(theta_min, theta_max + step_size, step_size), decimals=2).tolist()
        return theta_set


def work_models(theta, data, activity, case_id, timestamp, fallthroughs):
    copy.deepcopy(data)
    
    pn, im, fm = pm.discovery.discover_petri_net_inductive(
        log=data,
        noise_threshold=theta,
        multi_processing=False,
        activity_key=activity,
        case_id_key=case_id,
        timestamp_key=timestamp,
        disable_fallthroughs=fallthroughs
    )
    print(f'Model sucessfully mined for theta = {theta}')
    return pn, im, fm, theta


def work_metrics(model=tuple, data=pd.DataFrame, model_type=str) -> dict:
    copy.deepcopy(data)
    pn, im, fm, theta = copy.deepcopy(model)  

    metrics = OrderedDict()

    metrics['Noise_Threshold'] = theta

    if model_type == 'bpmn':
        metrics['Model'] = pm.convert_to_bpmn(pn, im, fm)
    elif model_type == 'tree':
        metrics['Model'] = pm.convert_to_process_tree(pn, im, fm)
    else:
        metrics['Model'] = pn


    net = pm.algo.evaluation.algorithm.apply(
        log=data, 
        net=pn,
        initial_marking=im,
        final_marking=fm
    )
    metrics.update(net)

    return metrics


def IMf(data        = pd.DataFrame,
        model_type  = str,
        export      = bool, 
        theta_min   = float,
        theta_max   = float,
        step_size   = float,
        case_id     = str,
        activity    = str,
        timestamp   = str,
        fallthroughs = bool) -> pd.DataFrame:
    
        if isinstance(data, pd.DataFrame) == False:
                hyperparam_metrics = pd.read_excel('DF\\HYP.xlsx', index_col=0)
                hyperparam_models  = None
        else:
                fallthroughs = not fallthroughs    
                k = theta_set(theta_min=theta_min, theta_max=theta_max, step_size=step_size) 
                num_cores = min(len(k), round(mp.cpu_count()/2))   

                with mp.Pool(processes=num_cores) as pool_a:
                        models = pool_a.starmap(
                                        work_models, 
                                        zip(k, 
                                        repeat(data), 
                                        repeat(activity), 
                                        repeat(case_id), 
                                        repeat(timestamp), 
                                        repeat(fallthroughs))
                        )
                                
                pool_a.close()
                pool_a.join()       

                with mp.Pool(processes=num_cores) as pool_b:
                        metrics = pool_b.starmap(
                                        work_metrics, 
                                        zip(models, 
                                        repeat(data),
                                        repeat(model_type))
                        )
                pool_b.close()
                pool_b.join()

                hyperparam_df   = pd.json_normalize(data=metrics[0], sep='_')                
        
                for i in metrics[1:]:
                        metrics_df = pd.json_normalize(data=i, sep='_')
                        hyperparam_df = pd.concat([hyperparam_df, metrics_df], ignore_index=True)
                
                hyperparam_df.set_index(keys='Noise_Threshold', drop=True, inplace=True)
                hyperparam_df = pd.concat([hyperparam_df.iloc[:, 0:4], hyperparam_df.iloc[:, -2:]], axis=1)

                hyperparam_models  = pd.DataFrame(data=hyperparam_df['Model'], index=hyperparam_df.index.to_list())
                hyperparam_metrics = hyperparam_df.drop(columns='Model', inplace=False)

                if export:
                        with pd.ExcelWriter('DF/HYP.xlsx') as writer:
                                hyperparam_metrics.to_excel(writer, sheet_name='HYP_P2P')
                        print('Dataframe successfully stored on disc as: HYP.xlsx')

                        for theta in hyperparam_models.index.to_list():
                               pm.write_bpmn(model=hyperparam_models.loc[theta, 'Model'],
                                             file_path=f'MODELS\\BPMN\\BPMN_{theta}')            

        return hyperparam_metrics, hyperparam_models


def IMf_Scores(data=pd.DataFrame, k=float) -> tuple:
        '''Insert pd.DataFrame as "data" and threshold "k" as float!
        
        Threshold k = filtering for a minimum amount of perfectly fitting traces of min. k-%:

        IMf_Scores computes a tuple containing:
        1) data = pd.Dataframe containing the metrics precision, simplicity, generalization, fitness
        and all 11 possible combination of the above 4 metrics such as P-G, P-F, S-G-F, etc
        2) id_max: Returns a nested dict. containing the 5 highest values for the 11 metrics and their indices
        3) ik_max: Returns a nested dict. same as id_max with respect to given k'''

        columns_abbr = list(data.columns) 
        columns_abbr[:4] = [col[0].upper() for col in columns_abbr[:4]]
        data = data.set_axis(columns_abbr, axis=1)

        # Column names
        c    = list(data.columns[:4])
        mask = data.iloc[:, -1] / 100 > k

        # Generate all comb (from 2 to 4 columns)
        sets = []
        for i in range(2, len(c) + 1):  # Start from 2 to avoid single-column comb
                sets.extend(itertools.combinations(c, i))

        ik_max = {}
        id_max = {}
        
        for comb in sets:
                divident = len(comb)
                new_column = '-'.join(comb)
                data[new_column] = data[list(comb)].sum(axis=1) / divident

                data_unique = data.sort_values(by=[new_column, 'Noise_Threshold'], ascending=True)
                data_unique = data_unique.drop_duplicates(subset=new_column, keep='last')

                ik_max[new_column] = data_unique.loc[mask, new_column].nlargest(5)
                id_max[new_column] = data_unique[new_column].nlargest(5)
                #top5.sort_values(axis=0, ascending=True, inplace=True)             

                
        return data, id_max, ik_max


def trace_filter(data=pd.DataFrame, 
                 theta=float, 
                 min_fitness= float, 
                 export=bool, 
                 activity=str, 
                 case_id=str, 
                 timestamp=str, 
                 fallthroughs=bool) -> pd.DataFrame:
        
        if isinstance(data, pd.DataFrame):
                if isinstance(theta, (float, int)):                
                        fallthroughs = not fallthroughs

                        pn, im, fm = pm.discovery.discover_petri_net_inductive(
                        log=data,
                        noise_threshold=theta,
                        multi_processing=True,
                        activity_key=activity,
                        case_id_key=case_id,
                        timestamp_key=timestamp,
                        disable_fallthroughs=fallthroughs
                        )
                        print(f'Model successfully mined for theta = {theta}\n')

                        diagnostics = pm.conformance.conformance_diagnostics_token_based_replay(
                        log=data,
                        petri_net=pn, 
                        initial_marking=im, 
                        final_marking=fm, 
                        activity_key=activity, 
                        timestamp_key=timestamp, 
                        case_id_key=case_id, 
                        return_diagnostics_dataframe=True
                        )
                        
                        #Delete artificially added 'START' and 'END' activity for export
                        #'START' and 'END' have been added to improve token-based replay performance 
                        #(unifiying creation- and sink-point for tokens)
                        data = data[~data[activity].isin(['START', 'END'])]

                        #Measure Dataframe-Size before filtering
                        num_cases_init      = len(data[case_id].unique())
                        num_activities_init = len(data[activity].unique())

                        #Filtering
                        diagnostics = diagnostics[diagnostics['trace_fitness'] >= min_fitness]
                        data = data[data[case_id].isin(diagnostics['case_id'])].reset_index(drop=True)
                        data = data.iloc[:, 0:3]

                        #Measure Dataframe-Size after filtering
                        num_cases_red      = len(data[case_id].unique())
                        num_activities_red = len(data[activity].unique())
                        red_rate_cases     = round(num_cases_red/num_cases_init * 100)
                        red_rate_activities= round(num_activities_red/num_activities_init * 100)     

                        print('Data-Log reduced from {} to {} Case-IDs:\nReduction-Rate = {}%\n'.format(num_cases_init, num_cases_red, red_rate_cases))
                        print('Data-Log reduced from {} to {} Activities:\nReduction-Rate = {}%\n'.format(num_activities_init, num_activities_red, red_rate_activities))

                else:
                        data = data[~data[activity].isin(['START', 'END'])]
                        data = data.iloc[:, 0:3]
                        data = format_timestamp(data=data, timestamp=timestamp)

                if export:
                        txt = 'P2PLog_{}_{}.csv'.format(timestmp, theta)
                        data.to_csv(path_or_buf=f'DF\\{txt}', index=False)
                        print(f'Dataframe successfully stored on disc as: {txt}\n')

                            
       
        return data
        

