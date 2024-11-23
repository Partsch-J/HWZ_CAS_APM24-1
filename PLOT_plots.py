import pm4py as pm
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pm4py.objects.bpmn.obj import BPMN


timestmp = dt.datetime.now().strftime("%y%m%d-%H-%M")

def round_to_base(x, base=int):
    return base * round(x/base)

   
def processmap(models=pd.DataFrame, graph=str, theta=dict, export=bool, variant=str, top_x=int) -> None:
    
    imp_xml = not isinstance(models, pd.DataFrame)

    #Checking if chosen top_x is greater than num_thetas => Chose smaller one
    top_x = min(top_x, len(next(iter(theta.values()))))    
    #Looping through theta-dictionary and getting the thetas corresponding to top_x values 
    #Values are already sorted desc by IMf_Scores output => first top_x values are highest!
    thetas = {t for k in theta.keys() for i, t in enumerate(theta[k].keys()) if i < top_x}

    if imp_xml:
        models = {}
        for t in thetas:
            bpmn = pm.read_bpmn(file_path=f'MODELS/BPMN/BPMN_{t}.bpmn')          
            models[t] = bpmn
        models = pd.DataFrame.from_dict(models, orient='index', columns=['Model'])
    else:
        models

    for t in thetas: 
        m = models.loc[t, 'Model']
        if not export:
            if graph == 'bpmn':
                pm.vis.view_bpmn(bpmn_graph=m)
            elif graph == 'tree':
                pm.vis.view_process_tree(tree=m)
            else:
                pm.vis.view_petri_net(petri_net=m)
        else:
            if graph == 'bpmn':
                num_act = len([node for node in m.get_nodes() if isinstance(node, BPMN.Activity)])
                pm.vis.save_vis_bpmn(
                    bpmn_graph=m,   
                    file_path=f'PLOTS/PLT_Processmaps/BPMN_{timestmp}_{variant}_{t}_{num_act}.svg'
                )
            elif graph == 'tree':
                pm.vis.save_vis_process_tree(
                    tree=m, 
                    file_path=f'PLOTS/PLT_Processmaps'
                )
            else:
                pm.vis.save_vis_petri_net(
                    petri_net=m, 
                    file_path=f'PLOTS/PLT_Processmaps'
                )



def heatmap(data: pd.DataFrame, name: str, color: str, export: bool, patches: dict, top_x: int) -> None:
    
    if isinstance(data, pd.DataFrame):
        k = data.iloc[:, -12]
        data = data.iloc[:, -11:]
        indices = data.index.tolist()
        
        color_acc_01 = '#5B9FC7'
        color_acc_02 = '#25597C'

        color = sns.light_palette(color, as_cmap=True, n_colors=10)
        sns.set_context("paper", font_scale=1.2, rc={"font.size": 9, "axes.titlesize": 15})

        plt.figure(figsize=(data.shape[1]*2, data.shape[0]))
        plot = sns.heatmap(data=data, 
                        annot=False, 
                        cbar=False, 
                        cmap=color, 
                        fmt=".5f", 
                        linewidths=1, 
                        linecolor='white')

        top_x = min(top_x, len(next(iter(patches.values()))))
        annot = range(0, round_to_base(x=data.shape[0], base=5), 5)
        annot_skip = []  
        max_thetas = set()

        for col in patches:
            i = 1
            pos = list(patches[col].keys())
            c = data.columns.get_loc(col)

            for pos in pos[:top_x]:
                p = data.index.get_loc(pos)
                max_thetas.add(p)
                annot_skip.append((c, p))

                patch_color = color_acc_02 if i == 1 else color_acc_01 
                plot.add_patch(Rectangle((c, p), 1, 1, fill=True, facecolor=patch_color, edgecolor='white', lw=1))
                plot.text(c + 0.5, p + 0.5, f'{round(k.iloc[p])}%', ha='center', va='center', color='white', fontdict={'fontweight': 'bold', 'fontsize': 20})
                i += 1

        # Set labels and title
        plt.xlabel('Metric Combinations')
        plt.ylabel('Noise Threshold (θ)')

        yticks = [i + 0.5 for i in annot]
        plot.set_yticks(yticks)
        ytick_labels = [f'{data.index[i]:.2f}' for i in annot]

        plot.set_yticklabels(ytick_labels, fontdict={'fontweight':'bold', 'fontsize':24}, rotation=0)    
        plot.set_xticklabels(plot.get_xticklabels(), fontdict={'fontweight':'bold', 'fontsize':24})

        i_min = min(indices)
        i_max = max(indices)
        k_min = round(k.iloc[list(max_thetas)].unique().min())
        plt.title(f'{name} ({i_min} < θ < {i_max} / k > {k_min}%)', loc="left", fontdict={"fontweight": "bold"}, y=1.0, pad=20)
        
        for c in range(data.shape[1]):
            for i in annot:
                if (c, i) not in annot_skip:
                    v = round(data.iloc[i, c], ndigits=3)
                    plot.text(c + 0.5, i + 0.5, f'{v}', ha='center', va='center', color='black', fontdict={'fontweight': 'bold', 'fontsize': 16})

        if export:
            plot.get_figure().savefig(f'PLOTS/PLT_Heatmaps/{timestmp}_{name}_Heatmap.svg')
        else:
            plt.show()
