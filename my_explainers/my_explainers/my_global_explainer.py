import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objects as go
import scipy as sp
import copy
import shap
from statistics import mode 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ipywidgets import interact, fixed, Dropdown, Layout, FloatSlider



def calculate_impacting_features(sample_shap_values):

    sample_shap_values.sort(key=lambda x: abs(x[1]), reverse=True)
    remaining_features = copy.deepcopy(sample_shap_values)
    impacting_features = list()
    
    for elem in sample_shap_values:
        
        impacting_features.append(elem)
        del remaining_features[0]

        sum_remaining_features = sum(v for name,v in remaining_features)
        sum_impacting_features = sum(v for name,v in impacting_features)

        if sum_impacting_features > 0 and sum_remaining_features > 0:
            break 

        if sum_impacting_features < 0 and sum_remaining_features < 0:
            break


    return [n for n,v in impacting_features]




 
def prepare_data(base_value, shap_values, classifier, features, encoded_X, X ,label, threshold):

    parcial_shap_values = shap_values #[label]
    

    data = []
    data_count = 0
    for s_values, f_values in zip(parcial_shap_values, X.iterrows()):
        tmp = dict()
        tmp["all_original"] = []
        tmp["pos"] = []
        tmp["neg"] = []
        feature_count = 0
        for s_v, f_v in zip(s_values, f_values[1]):
            feature_name = X.columns[feature_count]
            tmp["all_original"].append([feature_name, s_v, f_v, features.loc[data_count][feature_name], feature_count]) # [feature name, shap_value, feature_value, original feature value, orginal                                                                                                                             feature index]
            if s_v > 0:
                tmp["pos"].append([feature_name, s_v, f_v, features.loc[data_count][feature_name], feature_count])
            else:
                tmp["neg"].append([feature_name, s_v, f_v, features.loc[data_count][feature_name], feature_count])
            feature_count+=1
        
        tmp["pos"].sort(key=lambda x: abs(x[1]), reverse=True) 
        tmp["neg"].sort(key=lambda x: abs(x[1]), reverse=True)
        tmp["impacting_features"] = calculate_impacting_features([ (f[0], f[1]) for f in tmp["all_original"] ] )
        data.append(tmp)
        data_count += 1
        

    
    #compute predictions for all samples
    all_predictions_prob = [elem[label] for elem in classifier.predict_proba(encoded_X)] 
    all_predictions = [elem for elem in classifier.predict(encoded_X)] 



    #use following loop to compute min and max y coordinates in order to use as fences
    max_y = 1
    min_y = 0

    
    data_count = 0
    shap_order= shap.hclust_ordering(parcial_shap_values)


    #calculate coordinates for shap values
    for s in shap_order:
        sample=data[s]
        sample["prediction_prob"] = all_predictions_prob[s]
        sample["prediction"] = all_predictions[s]
        
        #calculate coordinates for negative shap values
        x = all_predictions_prob[s]
        for f in sample["neg"]:
            y = x - f[1]
            f.append(x)
            f.append(y)  # [feature name, shap_value, feature_value, original feature value, orginal feature index, x, y]

            if y > max_y:
                max_y = y

            x = y

        #calculate coordinates for positive shap values
        x = all_predictions_prob[s]
        for f in sample["pos"]:
            y = x - f[1]
            f.append(x) 
            f.append(y)  # [feature name, shap_value, feature_value, original feature value, orginal feature index, x, y]
            
            if y < min_y:
                min_y = y
            x=y

        data_count+=1



    #compute cluster matrix
    D = sp.spatial.distance.pdist(parcial_shap_values, "sqeuclidean")
    cluster_matrix = sp.cluster.hierarchy.complete(D)
    cluster_list = sp.cluster.hierarchy.fcluster(cluster_matrix, threshold, criterion="distance")

    
    #calculate cluster borders
    actual_cluster_order = []
    tmp_counter = 1
    tmp_current = cluster_list[shap_order[0]]
    for elem in shap_order:
        actual_cluster_order.append(cluster_list[elem])
        
        #get cluster number starting by 1
        if tmp_current == cluster_list[elem]:
            data[elem]["cluster"] = tmp_counter
        else:
            tmp_counter += 1
            tmp_current = cluster_list[elem]
            data[elem]["cluster"] = tmp_counter

        data[elem]["cluster_order_index"] = elem

    
    current_cluster = actual_cluster_order[0]
    cluster_borders = []
    for i in range(len(actual_cluster_order)):
        if current_cluster != actual_cluster_order[i]:
            cluster_borders.append(i)
            current_cluster = actual_cluster_order[i]
    
    cluster_borders = [0] + cluster_borders + [len(parcial_shap_values)]


    return data, max_y, min_y, shap_order, parcial_shap_values, all_predictions_prob, cluster_borders




 

def draw_plot(base_value, data, X, max_y, min_y, shap_order, parcial_shap_values, cluster_borders, current_cluster):
    
    traces = []    
    annotations = []
    shapes = []


    #draw shap values for each sample
    for data_count in range(len(shap_order)):
        sample=data[shap_order[data_count]]
        

        #draw negative shap values
        if len(sample["neg"]) != 0:
            tmp = ""
            for f in sample["neg"]:
                tmp += (str(f[0])) + " = " + (str(f[3])) + "<br>"

            trace = go.Scatter(x = [data_count, data_count],
                            y = [sample["neg"][0][-2], sample["neg"][-1][-1]],
                            hoveron = 'fills',
                            line_color = '#1E88E5',
                            showlegend= False,
                            mode = 'lines',
                            hoverinfo = "text",
                            text = f'<b>Sample id: {shap_order[data_count]}<br>Prediction: {sample["neg"][0][-2]}</b><br><br>{tmp}',
                            line = dict(
                                width = 1500/len(parcial_shap_values),
                                ),
                        )
            traces.append(trace)
        


        #draw positive shap values
        if len(sample["pos"]) != 0:

            tmp = ""
            for f in sample["pos"]:
                tmp += (str(f[0])) + " = " + (str(f[3])) + "<br>"

            trace = go.Scatter( x = [data_count, data_count],
                                y = [sample["pos"][0][-2], sample["pos"][-1][-1]],
                                hoveron = 'fills',
                                line_color = '#FF0D57',
                                showlegend= False,
                                mode = 'lines',
                                hoverinfo = "text",
                                text = tmp if (len(sample["neg"])!= 0) else f'<b>Sample id: {shap_order[data_count]}<br>Prediction: {sample["pos"][0][-2]}</b><br><br>{tmp}',
                                line = dict(
                                    width = 1500/len(parcial_shap_values),
                                    ),
                        )
            traces.append(trace)
       
   
    #get coordinates for white lines 
    pos_white = []
    neg_white = []
    for s in shap_order:
        sample=data[s]

        #for positive values
        tmp = []
        if len(sample["pos"]) == 0:
            tmp.append(sample["neg"][0][-2])
        for feature in sample["pos"]:
            tmp.append(feature[-2])
        if len(sample["pos"]) != 0:
            tmp.append(sample["pos"][-1][-1])
        pos_white.append(tmp)


        #for negative values
        tmp = []
        for feature in sample["neg"]:
            tmp.append(feature[-2])
        if len(sample["neg"]) != 0:
            tmp.append(sample["neg"][-1][-1])
        neg_white.append(tmp)
    
    




    


    #draw white lines for positive features
    for i in range(len(X.columns)):

        xxx = []
        yyy = []

        for count in range(len(pos_white)):
            sample = pos_white[count]

            if len(sample) == 0:
                if len(xxx) != 0:
                    trace = go.Scatter( x = xxx,
                                        y = yyy,
                                        hoveron = 'points+fills',
                                        line_color = '#FFFFFF',
                                        showlegend= False,
                                        mode = 'lines',
                                        hoverinfo= "none",
                                        line = dict(
                                        width = 0.5,
                                        ),
                                    )
                    traces.append(trace)
                    xxx = []
                    yyy = []
            
            else:
                new_y = sample.pop(0)
                yyy.append(new_y)
                xxx.append(count)

        trace = go.Scatter( x = xxx,
                            y = yyy,
                            hoveron = 'points+fills',
                            line_color = '#FFFFFF',
                            showlegend= False,
                            mode = 'lines',
                            hoverinfo= "none",
                            line = dict(width = 0.5,
                                        ),
                                    )
        traces.append(trace)






    #draw white lines for negative features
    for i in range(len(X.columns)):

        xxx = []
        yyy = []

        for count in range(len(neg_white)):
            sample = neg_white[count]

            if len(sample) == 0:
                if len(xxx) != 0:
                    trace = go.Scatter( x = xxx,
                                        y = yyy,
                                        hoveron = 'points+fills',
                                        line_color = '#FFFFFF',
                                        showlegend= False,
                                        mode = 'lines',
                                        hoverinfo= "none",
                                        line = dict(
                                        width = 0.5,
                                        ),
                                    )
                    traces.append(trace)
                    xxx = []
                    yyy = []
            
            else:
                new_y = sample.pop(0)
                yyy.append(new_y)
                xxx.append(count)

        trace = go.Scatter( x = xxx,
                            y = yyy,
                            hoveron = 'points+fills',
                            line_color = '#FFFFFF',
                            showlegend= False,
                            mode = 'lines',
                            hoverinfo= "none",
                            line = dict(width = 0.5,
                                        ),
                                    )
        traces.append(trace)




    
   
    for i in range(len(cluster_borders)):
        if i < ((len(cluster_borders))-1):

            #draw rectangles around clusters
            shape = go.layout.Shape(type='rect',
                                x0= cluster_borders[i],
                                x1= cluster_borders[i+1],
                                y0= max_y + 0.05, 
                                y1= min_y - 0.2,
                                fillcolor = "rgba(191,191,191,0)" if current_cluster == (i+1) else "rgba(191,191,191,0.35)",
                                line=dict(
                                        color="#70AD47",
                                        width= 6 if current_cluster == (i+1) else 2,
                                        )
                                )
            shapes.append(shape)

            #draw numbers of clusters
            annotation = go.layout.Annotation(
                                x = cluster_borders[i] + ((cluster_borders[i+1] - cluster_borders[i])/2),
                                y = min_y - 0.12,
                                text = i+1,
                                showarrow = False,
                                font=go.layout.annotation.Font(
                                    color = '#8F8F8F',
                                    size = 15
                                    ),
                                visible = True,
                                )   
            annotations.append(annotation)




    
    #design layout of plot
    layout = go.Layout(
        
        hovermode='x', 
        plot_bgcolor = '#FFFFFF',
        autosize=False,
        height = 500,
        width = 1500,
        margin= dict(
                t = 70, 
                b = 40,
                        ),
        legend = dict(
                y = 0.8,
                font= dict(
                    color = '#5E5E5E',
                    size = 10,
                    ),
        ),
        xaxis = go.layout.XAxis(
            showgrid = False,
            fixedrange = True,
            range= [0, len(parcial_shap_values)],
            showticklabels = True,
            tickcolor = '#A6A6A6',
            tickfont = go.layout.xaxis.Tickfont(
                color = '#A6A6A6',
            ),
            side= "top",
            ticks = "outside",
            zeroline = False,
            showline = False,
            linecolor = '#A6A6A6',
        
        ),
        yaxis = go.layout.YAxis(
            showgrid = False,
            showticklabels = True,
            tickcolor = '#A6A6A6',
            tickfont = go.layout.yaxis.Tickfont(
                color = '#A6A6A6',
            ),
            tick0 = base_value,
            tickformat = '.3f',
            fixedrange = True,
            range = [min_y-0.3, max_y+0.1],
            zeroline = False,
            showline = True,
            linecolor = '#A6A6A6',
            dtick = 0.2,
            ticks = "outside"
        ),
        
        annotations = annotations,
        shapes = shapes,
    )
    

    #create figure
    fig = go.Figure(data = traces, layout = layout)

    #display plot
    py.iplot(fig)





 
def draw_table(data, current_cluster):

   number_observations = 0
   cluster_output_values = []
   cluster_predictions = []
   
   for sample in data:
     if sample["cluster"] == current_cluster:
         number_observations+=1
         cluster_output_values.append(sample["prediction_prob"])
         cluster_predictions.append(sample["prediction"])

   avg_value = 0 if len(cluster_output_values) == 0 else sum(cluster_output_values)/len(cluster_output_values)
    
   fig = go.Figure(data=[go.Table(
            header=dict(values = ["cluster number", "prediction", "number of observations", "min output value", "max output value", "average output value"],
                        height = 40,
                        font=dict(
                            size= 15,
                            color= '#FFFFFF',
                        ),
                        fill=dict(
                            color= '#BFBFBF',
                        ),
                        line=dict(
                            width= 2,
                        ),
                ),
            cells=dict(values =[[current_cluster], [mode(cluster_predictions)], [number_observations], 
                                 ["{:.3f}".format(min(cluster_output_values))], ["{:.3f}".format(max(cluster_output_values))], ["{:.3f}".format(avg_value)],
                                ],
                        height = 30,
                        font=dict(
                            size= 15,
                            color= '#5E5E5E'
                        ),
                        fill=dict(
                           color= 'rgba(191,191,191,0.3)',
                        ),
                        line=dict(
                            width= 2,
                        ),
            ),
            visible= True,
            columnwidth = [(1/6) * 6],
            
   )])
   
   fig.update_layout(width=1500, 
                    height=140, 
                    margin= dict(
                            t = 40,
                            b = 30,
                            l = 0,
                            r = 0
                        )
                    )

   fig.show()

   
    
    

def get_statistics(data, current_cluster, feature_name):
    
    positive_cases = 0
    negative_cases = 0
    impacting = 0
    not_impacting = 0
    original_feature_values = []
    for sample in data:
        if sample["cluster"] == current_cluster:
            for f in sample["all_original"]:
                if f[0]==feature_name:
                    original_feature_values.append(f[3])
                    if feature_name in sample["impacting_features"]:
                        impacting += 1
                    else:
                        not_impacting +=1
                    if f[1] > 0: 
                        positive_cases +=1 
                    else:
                        negative_cases +=1

    return [feature_name, positive_cases, negative_cases, impacting, not_impacting, original_feature_values]








 
def draw_statistics(features, current_cluster, feature_name, number, positive_cases, negative_cases, impacting, not_impacting, original_feature_values):


    annotations = []
    
    #display number of feature
    annotation = go.layout.Annotation(
        x = 0.05,
        y = 0.5,
        text = number,
        showarrow = False,
        font=go.layout.annotation.Font(
            color = '#5E5E5E',
            size = 20
            ),
        xref = "paper",
        yref = "paper",

    )   
    annotations.append(annotation)

    #display feature name

    annotation = go.layout.Annotation(
        x = 0.2,
        y = 0.5,
        text = feature_name,
        showarrow = False,
        font=go.layout.annotation.Font(
            color = '#5E5E5E',
            size = 20
            ),
        xref = "paper",
        yref = "paper",
        xanchor = "center",

    )   
    annotations.append(annotation)




    shapes = []

    #draw white vertical separator lines for table
    separators_x = [0.1, 0.3, 0.5, 0.7]

    for elem in separators_x:
    
        shape = go.layout.Shape(type='line',
                                x0 = elem,
                                x1 = elem,
                                y0 = 0,
                                y1 = 1,
                                yref = 'paper',
                                xref = "paper",
                                line=dict(
                                        color= '#FFFFFF',
                                        width=3,
                                        )
                                )
        shapes.append(shape)



    #draw white horizontal separator lines for table
    shape = go.layout.Shape(type='line',
                                x0 = 0,
                                x1 = 1,
                                y0 = -0.1,
                                y1 = -0.1,
                                yref = 'paper',
                                xref = "paper",
                                line=dict(
                                        color= '#FFFFFF',
                                        width=3,
                                        )
                                )
    shapes.append(shape)





    
    traces = []

    #draw pie chart displaying amount of positive and negative SHAP values within cluster
    trace = go.Pie(labels=["positive", "negative"], 
                    values=[positive_cases, negative_cases],
                    domain = {'x': [0.3, 0.5], 'y': [0, 1]},
                    showlegend=False,
                    textposition = "none",
                    hoverinfo = "percent+label",
                    hole= 0.5,
                    marker = dict(colors = ['#FF0D57','#1E88E5'],
                                  line = dict(color='#FFFFFF',
                                              width = 2
                                             )

                                    )
        )
    traces.append(trace)


    #draw pie chart displaying for how many of the samples withing this cluster the current feature is impacting
    trace = go.Pie(labels=["impacting", "not impacting"], 
                    values=[impacting, not_impacting],
                    domain = {'x': [0.5, 0.7], 'y': [0, 1]},
                    showlegend=False,
                    textposition = "none",
                    hoverinfo = "percent+label",
                    hole= 0.5,
                    marker = dict(colors = ['#548235','#ED7D31'],
                                  line = dict(color='#FFFFFF',
                                              width = 2
                                             )

                                    )
        )
    traces.append(trace)
    

    
  
    # draw distplot for feature values for this feature for whole dataset
    trace = go.Histogram(x=features.loc[:,feature_name],
                        opacity = 0.4,
                        marker = dict(color = "#ED7D31",
                        ),
                        histnorm = "percent",
                        hoverinfo = "x+y",
                        showlegend=False,
            )
    traces.append(trace)


    # draw distplot for feature values for this feature for current cluster
    trace = go.Histogram(x=original_feature_values,
                        opacity = 0.4,
                        marker = dict(color = "#4472C4",
                        ),
                        histnorm = "percent",
                        hoverinfo = "x+y",
                        showlegend=False,
            )
    traces.append(trace)

    
    # design layout
    layout = go.Layout(width=1500, 
                       height=200, 
                       autosize = False,
                       barmode='overlay',
                       paper_bgcolor = 'rgba(191,191,191,0.3)',
                       plot_bgcolor = 'rgba(191,191,191,0.0)',
                       margin= dict(t = 20,
                                    b = 20,
                                    l = 0,
                                    r = 0
                                    ),
                        xaxis = go.layout.XAxis(
                                        showgrid = False,
                                        tickcolor = '#A6A6A6',
                                        tickfont = go.layout.xaxis.Tickfont(
                                            color = '#A6A6A6',
                                            size = 10

                                        ),
                                        side= "bottom",
                                        ticks = "outside",
                                        domain = [0.73, 0.99], 
                                        zeroline = False,
                                        showline = False,
                                        ),
                        yaxis = go.layout.YAxis(
                                        tickcolor = '#A6A6A6',
                                        tickfont = go.layout.yaxis.Tickfont(
                                            color = '#A6A6A6',
                                            size = 10
                                        ),
                                        ticks = "outside",
                                        domain = [0.04, 0.98], 
                                        zeroline = False,
                                        showline = False,
                                        ),
                        annotations = annotations,
                        shapes = shapes,
                        )

    fig = go.Figure(traces, layout)

    fig.show()



def draw_header():
    
    fig = go.Figure(data=[go.Table(
            header=dict(values = ["order by impact in current cluster", "feature", "cases in which positive SHAP value<br>cases in which negative SHAP value", 
                                    "cases in which impacting feature<br>cases in which <i>not</i> impacting feature",  
                                    "feature value distibution for current cluster<br>feature value distibution for whole dataset"],
                        height = 50,
                        align='center',
                        font=dict(
                            size= 13,
                            color= '#FFFFFF',
                        ),
                        fill=dict(
                            color= '#BFBFBF',
                        ),
                        line=dict(
                            width= 3,
                        )
                ),
            visible= True,
            columnwidth = [0.1, 0.2, 0.2, 0.2, 0.3],
            
    )])

    shapes = []


    circles = [('#FF0D57', 0.312, 0.6, 0.317, 0.74), ('#1E88E5',0.312, 0.3, 0.317,0.44), 
                ('#ED7D31', 0.507, 0.3, 0.512, 0.44), ('#548235', 0.507, 0.6, 0.512, 0.74),
                ("#4472C4", 0.746, 0.6, 0.751, 0.74), ('#ED7D31', 0.746, 0.3, 0.751,0.44)]


    for elem in circles:

            shape = go.layout.Shape(type='circle',
                                        xref="paper",
                                        yref="paper",
                                        fillcolor=elem[0],
                                        line_color=elem[0],
                                        x0=elem[1],
                                        y0=elem[2],
                                        x1=elem[3],
                                        y1=elem[4],
                                        )
            shapes.append(shape)


    
    fig.update_layout(width=1500, 
                        height=55, 
                        margin= dict(
                                t = 0,
                                b = 0,
                                l = 0,
                                r = 0
                            ),
                        shapes = shapes
                        )

    fig.show()





def draw_all_elements(base_value, features, data, X, max_y, min_y, shap_order, parcial_shap_values, cluster_borders, current_cluster):

    
    #draw table
    draw_table(data, current_cluster)

    #draw plot
    draw_plot(base_value, data, X, max_y, min_y, shap_order, parcial_shap_values, cluster_borders, current_cluster)

    #draw header 
    draw_header()

    #get statistics
    statistics = []
    for elem in data[0]["all_original"]:
        statistics.append(get_statistics(data, current_cluster, elem[0]))
        
    statistics.sort(key=lambda x: abs(x[3]), reverse=True)
    
    #draw statistics
    counter = 1
    for elem in statistics:
        draw_statistics(features, current_cluster, elem[0], counter, elem[1], elem[2], elem[3], elem[4], elem[5])
        counter +=1
    






def update_view(base_value, shap_values, classifier, features, X, label, threshold, decoder):

    df_shap_values = pd.DataFrame(shap_values, columns = X.columns) 
    decoded_X = copy.deepcopy(X)

    for feature in decoder:
        sum_shap = [0] * len(shap_values)

        for elem in decoder[feature]:
            for i in range(len(shap_values)):
                sum_shap[i] += df_shap_values[elem][i]
                
            df_shap_values = df_shap_values.drop([elem], axis=1)
            decoded_X = decoded_X.drop([elem], axis=1)
        
        df_shap_values[feature] = sum_shap
        decoded_X[feature] = features[feature] 

    df_shap_values = df_shap_values[features.columns]
    decoded_X = decoded_X[features.columns]

    shap_values = df_shap_values.to_numpy()

    #get result of first widget -> prepared data
    data, max_y, min_y, shap_order, parcial_shap_values, all_predictions_prob, cluster_borders = prepare_data(base_value, shap_values, classifier, features, X, decoded_X, label, threshold)

    interact(draw_all_elements, base_value = fixed(base_value), features = fixed(features), data = fixed(data), X=fixed(X), max_y = fixed(max_y), min_y = fixed(min_y), shap_order = fixed(shap_order), 
            parcial_shap_values = fixed(parcial_shap_values), cluster_borders = fixed(cluster_borders),
            current_cluster = Dropdown(
                            options=[e+1 for e in range(len(cluster_borders)-1)],
                            value=1,
                            description='Currently selected cluster:',
                            disabled=False,
                            layout=Layout(width='99%', padding = "0px 650px 30px 400px"),
                            style = {'description_width': 'initial'})
            ) 


    




"""
 Parameters:
 -----------
 base value : float
    as computed by SHAP

 shap values : numpy.array
    as computed by SHAP

 classifier

 features : dataframe
   matrix of orginal feature values (# samples x # features)

 X : dataframe
    matrix of feature values as used to train the classifier (# samples x # features)

 label: int
    label of class the plots should be for

 decoder: dict
   original column names as keys and a list with new column names as values

 """

def my_global_explainer(base_value, shap_values, classifier, features, X, label, decoder=dict()):
    
    #create widget for threshold
    slider = FloatSlider(value=1,
                                min=0,
                                max=2,
                                step=0.025,
                                description='Threshold for cluster formation:',
                                continuous_update=False,
                                orientation='horizontal',
                                layout=Layout(width='99%', padding = "40px 600px 20px 400px"),
                                style = {'description_width': 'initial'}
                            )

    interact(update_view, base_value = fixed(base_value), shap_values= fixed(shap_values), classifier=fixed(classifier), features=fixed(features), X=fixed(X), 
                    label=fixed(label), threshold=slider, decoder = fixed(decoder))
    
    

def get_clusters(base_value, shap_values, classifier, features, X, label, threshold, decoder=dict()):

    df_shap_values = pd.DataFrame(shap_values, columns = X.columns) 
    decoded_X = copy.deepcopy(X)

    for feature in decoder:
        sum_shap = [0] * len(shap_values)

        for elem in decoder[feature]:
            for i in range(len(shap_values)):
                sum_shap[i] += df_shap_values[elem][i]
                    
            df_shap_values = df_shap_values.drop([elem], axis=1)
            decoded_X = decoded_X.drop([elem], axis=1)
            
        df_shap_values[feature] = sum_shap
        decoded_X[feature] = features[feature] 

    df_shap_values = df_shap_values[features.columns]
    decoded_X = decoded_X[features.columns]

    shap_values = df_shap_values.to_numpy()

    #get result of first widget -> prepared data
    data, max_y, min_y, shap_order, parcial_shap_values, all_predictions_prob, cluster_borders = prepare_data(base_value, shap_values, classifier, features, X, decoded_X, label, threshold)

    cluster_data = []
    for current_cluster in range(1, len(cluster_borders)):

        cluster = {} 
        counter = 0
        
        for sample in data:
            if sample["cluster"] == current_cluster:
                cluster[shap_order[counter]] = (features.loc[shap_order[counter]])
                counter +=1

        cluster_data.append(pd.DataFrame(cluster).T)

    return cluster_data















    

