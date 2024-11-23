from IMP_data import csv_to_log 
from PERF_hyperparmTuning import IMf, IMf_Scores, trace_filter
from PLOT_plots import heatmap, processmap

if __name__ == '__main__':

    df_P2P_red  = "DF\\DF_P2P_red.csv"
    df_P2P      = "DF\\DF_P2P.csv"


    case_id     = '%KEY_CASEID'
    activity    = 'P2P_ActionCode'
    timestamp   = 'P2P_TimeStamp'

    P2PLog = csv_to_log(filepath=df_P2P, 
                    column_sep=';',
                    case_id=  case_id,
                    activity= activity,
                    timestamp=timestamp)
    


    df_input = None
    exp      = True
    top      = 3


    P2PLog_Red = trace_filter(data=df_input, 
                            theta=0.15, 
                            min_fitness=1, 
                            export=exp, 
                            activity=activity, 
                            case_id=case_id, 
                            timestamp=timestamp, 
                            fallthroughs=True)
   

    metrics, models = IMf(data=df_input,
                        export=exp,
                        model_type= 'bpmn',
                        step_size= 0.01,
                        case_id=case_id,
                        activity=activity,
                        timestamp=timestamp,
                        fallthroughs=True)
        


    data, id_max, ik_max = IMf_Scores(data=metrics, k=0.7)


    
    heatmap(data=None, 
            color='gray', 
            export=exp, 
            name='HYP', 
            patches=id_max, 
            top_x=top)


    processmap(models=models, 
               theta=ik_max, 
               graph='bpmn', 
               export=exp, 
               variant='V1', 
               top_x=top)

    
