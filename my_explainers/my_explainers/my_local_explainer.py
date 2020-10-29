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
from ipywidgets import interact, fixed, Dropdown, Layout, FloatSlider, BoundedIntText, HBox, VBox


class My_Local_Explainer:

    def __init__(self, base_value, shap_values, classifier, features, X, encoded_X, sample_id, label):
        
        self.base_value = base_value
        self.shap_values = shap_values
        self.label = label
        self.parcial_shap_values = shap_values #[self.label]
        self.features = features
        self.X = X
        self.encoded_X = encoded_X
        self.sample_id = sample_id
        self.classifier = classifier

        # get predictions for all samples and extract current one
        self.all_predictions = self.classifier.predict(self.encoded_X)
        self.sample_prediction = self.all_predictions[self.sample_id]

        # get prediction propabilities for all samples and extract current one
        all_predictions_prob = self.classifier.predict_proba(self.encoded_X)
        self.all_predictions_prob = [elem[self.label] for elem in all_predictions_prob] 
        self.sample_prediction_prob = self.all_predictions_prob[self.sample_id]

        #compute list of lists of shap values as follows (feature name, shap value)
        self.list_shap_values=list()

        #use following loop to also compute the max positive/negative shap values for whole dataset
        self.max_pos_shap_value = ("", -1, 0)
        self.max_neg_shap_value = ("", 1, 0)
        self.sample_count=0

        for sample in self.parcial_shap_values:
            current_shap_values = list()
            for n, v in zip(self.X.columns, sample):
                current_shap_values.append([n,v])
                if v > self.max_pos_shap_value[1]:
                    self.max_pos_shap_value=(n,v, self.sample_count)
                if v < self.max_neg_shap_value[1]:
                    self.max_neg_shap_value=(n,v, self.sample_count)
            current_shap_values.sort(key=lambda x: abs(x[1]), reverse=True)
            self.sample_count+=1
            self.list_shap_values.append(current_shap_values)


        #get shap values values for current samplesample
        self.sample_shap_values = copy.deepcopy(self.list_shap_values[self.sample_id])
        self.sample_feature_values = self.features.loc[self.sample_id]
       
        #calculate impacting features and get impacting features for current sample
        self.new_calculate_impacting_features()
        self.sample_impacting_features = self.list_impacting_features[self.sample_id]
        self.sample_impacting_features_names = [elem[0] for elem in self.sample_impacting_features]


        #colors from SHAP in hsv format to help compute color gradients
        self.red = (342,94.9,100)
        self.blue=(208,86.9,89.8)
        self.red_colors = [self.red]
        tmp_red=self.red[1]
        self.blue_colors = [self.blue]
        tmp_blue=self.blue[1]
        
        for i in range(5):
            tmp_red = tmp_red-15
            tmp_blue = tmp_blue-15
            self.red_colors.append((self.red[0], tmp_red, self.red[2]))
            self.blue_colors.append((self.blue[0], tmp_blue, self.blue[2]))
        

        # create an evaluation for impacting features for all samples
        self.eval_impacting_features=dict()

        for feature in self.X.columns:
            self.eval_impacting_features[feature] = (0, 0)

        for elem in self.list_impacting_features:
            if len(elem) == 1:
                (a,b) = self.eval_impacting_features[elem[0][0]] 
                self.eval_impacting_features[elem[0][0]] = (a+1, b+1)
            else:
                for e in elem:
                    (a,b) = self.eval_impacting_features[e[0]]
                    self.eval_impacting_features[e[0]] = (a+1, b)

        for item in self.eval_impacting_features.items():
            (a,b) = item[1]
            self.eval_impacting_features[item[0]] = ((a/self.sample_count)*100, (b/self.sample_count)*100)

    
        #get different classes
        self.possible_predictions = set(self.all_predictions)
        
        #create dict() where: class-> [output values]
        self.output_values_by_prediction = dict()

        for pred in self.possible_predictions:
            self.output_values_by_prediction[pred]=[]
            for p, prob in zip(self.all_predictions, self.all_predictions_prob):
                if p == pred:
                    self.output_values_by_prediction[pred].append(prob)
        

        #for label and not_label get min, max, q1, q1, median and create box plots
        df_label = pd.DataFrame(self.output_values_by_prediction[label])
        df_not_label = []
        for pred in self.possible_predictions:
            if pred != label:
                df_not_label = df_not_label + self.output_values_by_prediction[pred]
        self.box_plots = [df_label, pd.DataFrame(df_not_label)]

        #min and max for box_plots
        min_box = min(float((self.box_plots[0]).min()),float((self.box_plots[1]).min()))
        max_box = max(float((self.box_plots[0]).max()),float((self.box_plots[1]).max()))

        #calculate x coordinates for polygons
        self.pos_x_coordinates, self.neg_x_coordinates = self.calculate_x_coordinates()

        # get max and min x coordinates of polygons and use them to calculate fences for x axis
        if not len(self.pos_x_coordinates) == 0:
            self.min_x_coordinate = min(self.pos_x_coordinates[-1][-2][-1], self.sample_prediction_prob - self.max_pos_shap_value[1], self.base_value, self.sample_prediction_prob, min_box)
        else:
            self.min_x_coordinate = min(self.sample_prediction_prob, self.sample_prediction_prob - self.max_pos_shap_value[1], min_box)

        if not len(self.neg_x_coordinates) == 0:
            self.max_x_coordinate = max(self.neg_x_coordinates[-1][-2][-1], self.sample_prediction_prob - self.max_neg_shap_value[1], self.base_value, self.sample_prediction_prob, max_box)
        else:
            self.max_x_coordinate = max(self.sample_prediction_prob, self.sample_prediction_prob - self.max_neg_shap_value[1], max_box)



        #get numerical features -> just numbers
        self.my_numerical_features = []
        for feature in self.features:
            if not isinstance(self.features[feature][0], str):
                if isinstance((self.features[feature][0]).item(), int) or isinstance(self.features[feature][0], float):
                    self.my_numerical_features.append(feature)
        
        #get min, max and average value of numerical features over all samples
        self.feature_value_range = dict()
        for f in self.my_numerical_features:
            self.feature_value_range[f] = ["{:.1f}".format(np.amin(self.features[f])), "{:.1f}".format(np.average(self.features[f])), "{:.1f}".format(np.amax(self.features[f]))]

        #add feature values to list_shap_values
        for sv, fv in zip(self.list_shap_values, self.features.iterrows()):
            for i in range(len(sv)):
                sv[i].append(fv[1][sv[i][0]])


            
    
    
    def calculate_feature_value_range(self, feature):

        s_values_pos = []
        s_values_neg = []
        feature_values =[]
        intervall_index = 0

        for sample in self.list_shap_values:
            for f in sample:
                if f[0] == feature:
                    feature_values.append(f[2])
                    #if f[2] == self.sample_feature_values[feature]:
                        #intervall_index = count


        # do for features with less then 11 feature values
        if (len(set(self.features[feature])) <= 10) or feature not in self.my_numerical_features:
            
            for sample in self.list_shap_values:
                for f in sample:
                    if f[0] == feature and f[2] == self.sample_feature_values[feature]:
                        if f[1] > 0:
                            s_values_pos.append(f[1]) 
                        else: 
                            s_values_neg.append(f[1])
            
            feature_value_intervall = None



        # do for all other features
        else:
            
            feature_value_intervall = (pd.cut(np.array(feature_values), 10, retbins=True))[0] #[intervall_index]

            for i in feature_value_intervall:
                if self.sample_feature_values[feature] in i:
                    feature_value_intervall = i
                    break


            for sample in self.list_shap_values:
                for f in sample:
                    if f[0] == feature and f[2] in feature_value_intervall:
                        if f[1] > 0:
                            s_values_pos.append(f[1]) 
                        else: 
                            s_values_neg.append(f[1])
            
        avg_pos = (sum(s_values_pos) / len(s_values_pos)) if (len(s_values_pos) != 0) else 0
        avg_neg = (sum(s_values_neg) / len(s_values_neg)) if (len(s_values_neg) != 0) else 0
        
        return avg_pos, avg_neg, len(s_values_pos), len(s_values_neg), feature_value_intervall
        
    






    def new_calculate_impacting_features(self):

        self.list_impacting_features = list()
        threshold = 0.5
        
        for sample in self.list_shap_values:

            remaining_features = copy.deepcopy(sample)
            impacting_features = list()
            sum_all = sum(v for name,v in sample)
            sum_all = self.base_value - threshold + sum_all 

            for e in range(len(sample)):
                
                impacting_features.append(sample[e])
                del remaining_features[0]

                sum_impacting_features = sum(v for name,v in impacting_features)
                current_sum = self.base_value - threshold + sum_impacting_features
                
                if current_sum > 0 and sum_all > 0:
                    break 
                if current_sum < 0 and sum_all < 0:
                    break


            self.list_impacting_features.append(impacting_features)




  

    def calculate_x_coordinates(self):

        #split SHAP values in positive and negative ones
        pos_shap = []
        neg_shap = []
        count = 0
    
        for elem in self.sample_shap_values:
            if elem[1] > 0:
                pos_shap.append(elem)
            else:
                neg_shap.append(elem)
            elem.append(count)
            count +=1

        # calculate x coordinates for polygons of features with positive SHAP values
        self.pos_x_coordinates = []
        x1 = 0
        x2 = self.sample_prediction_prob

        for elem in pos_shap:
            x3 = x2 - elem[1] + 0.01
            x4 = x3 - 0.01
            self.pos_x_coordinates.append([elem[0], elem[1],[x1,x2,x3,x4], elem[2]])
            x1 = x3
            x2 = x4

        # calculate x coordinates for polygons of features with negative SHAP values
        self.neg_x_coordinates = []
        x1 = 0
        x2 = self.sample_prediction_prob

        for elem in neg_shap:
            x3 = x2 - elem[1] - 0.01
            x4 = x3 + 0.01
            self.neg_x_coordinates.append([elem[0], elem[1],[x1,x2,x3,x4], elem [2]])
            x1 = x3
            x2 = x4
        
        return self.pos_x_coordinates, self.neg_x_coordinates, 









    def draw_plot(self, highlighted_feature, boxplots):

        traces = []    
        annotations = []
        shapes = []


        # marker for max pos shap values over all samples
        annotation = go.layout.Annotation(
            x = self.sample_prediction_prob - self.max_pos_shap_value[1] - 0.03,
            y = 0.85,
            text = '|max| ↑',
            showarrow = False,
            font=dict(color = '#FF0D57',
                      size = 11
                     ),
        )   
        annotations.append(annotation)


        # marker for neg pos shap values over all samples
        annotation = go.layout.Annotation(
            x = self.sample_prediction_prob - self.max_neg_shap_value[1] + 0.03,
            y = 0.85,
            text = '|max| ↓',
            showarrow = False,
            font=dict(color = '#1E88E5',
                      size = 11
            ),
        )   
        annotations.append(annotation)


        # marker "higher"
        annotation = go.layout.Annotation(
            text = 'higher ⇥',
            font= dict(color = '#FF0D57',
                       size = 11
                       ),
            showarrow = False,
            x = self.sample_prediction_prob,
            y = 1.35,
            yref = 'paper',
            xanchor = 'right',
            yanchor = 'bottom',
        )
        annotations.append(annotation)


        # marker "lower"
        annotation = go.layout.Annotation(
            text = '⇤ lower',
            font= dict(color = '#1E88E5',
                       size = 11
                      ),
            showarrow = False,
            x = self.sample_prediction_prob,
            y = 1.35,
            yref = 'paper',
            xanchor = 'left',
            yanchor = 'bottom',
        )
        annotations.append(annotation)


        # base value annotation
        annotation = go.layout.Annotation(
            text = 'base value',
            x = self.base_value,
            y = 1.2,
            yref = 'paper',
            showarrow = False,
            yanchor = 'bottom',
            font = go.layout.annotation.Font(
                color = '#5E5E5E',

            ),
        )
        annotations.append(annotation)


        # output value annotation
        annotation = go.layout.Annotation(
            text = 'output value',
            x = self.sample_prediction_prob,
            y = 1.2,
            yref = 'paper',
            showarrow = False,
            yanchor = 'bottom',
            font = go.layout.annotation.Font(
                color = '#5E5E5E',

            ),
        )
        annotations.append(annotation)


        # display output value
        annotation = go.layout.Annotation(
            text = f'<b>{self.sample_prediction_prob:.3f}</b>',
            x = self.sample_prediction_prob,
            y = 1,
            showarrow = False,
            xanchor='center',
            yref='paper',
            yanchor = 'bottom',
            bgcolor = '#FFFFFF',
            borderpad = 5,
            font = go.layout.annotation.Font(
                color = '#5E5E5E',

            ),
        )
        annotations.append(annotation)


        # display base value
        annotation = go.layout.Annotation(
            text = f'<b>{self.base_value:.3f}</b>',
            x = self.base_value,
            y = 1,
            showarrow = False,
            xanchor='center',
            yref='paper',
            yanchor = 'bottom',
            bgcolor = '#FFFFFF',
            borderpad = 5,
            font = go.layout.annotation.Font(
                color = '#5E5E5E',

            ),
        )
        annotations.append(annotation)



        #display prediction
        annotation = go.layout.Annotation(
            text = f'prediction: label {self.sample_prediction}',
            x = 0,
            y = 2.4,
            font = go.layout.annotation.Font(
                color = '#5E5E5E',
                size = 15,
            ),
            showarrow = False,
            xref = 'paper',
            yref = 'paper'
        )
        annotations.append(annotation)


        #display regressor
        """annotation = go.layout.Annotation(
            text = f'regressor: {self.label}',
            x = 0,
            y = 2.1,
            font = go.layout.annotation.Font(
                color = '#5E5E5E',
                size = 15,
            ),
            showarrow = False,
            xref = 'paper',
            yref = 'paper'
        )
        annotations.append(annotation)"""


        # draw vertical line at output value
        shape = go.layout.Shape(type='line',
                            x0=self.sample_prediction_prob,
                            x1=self.sample_prediction_prob,
                            y0= 0,
                            y1= 1.1,
                            yref = "paper",
                            line=dict(
                                    color="#A6A6A6",
                                    width=4,
                                    )
                            )
        shapes.append(shape)


        #draw bosplots if selected
        if boxplots:

            self.box_plot_shapes = []
            self.box_plot_colors = [("#FF9300", "rgba(255,147,0, 0.3)"),("#70AD47", "rgba(112,173,71, 0.3)")]
            
            for elem, c in zip(self.box_plots, self.box_plot_colors):
            
                q1 = elem.quantile(0.25)
                q2 = elem.quantile(0.75)
                median = elem.quantile(0.5)
                minimum = elem.min()
                maximum = elem.max()
                box_plot_values = [minimum, median, maximum]
                
                
                for e in box_plot_values:
                    shape = go.layout.Shape(type='line',
                                    x0= float(e),
                                    x1= float(e),
                                    y0= 1.6,
                                    y1= 1.8,
                                    yref = 'paper',
                                    line=dict(
                                            color=c[0],
                                            width=3,
                                            )
                                    )
                    shapes.append(shape)


                shape = go.layout.Shape(type='line',
                                    x0 = float(minimum),
                                    x1= float(q1),
                                    y0= 1.7,
                                    y1= 1.7,
                                    yref = 'paper',
                                    line=dict(
                                            color=c[0], 
                                            width=3,
                                            )
                                    )
                shapes.append(shape)



                shape = go.layout.Shape(type='line',
                                    x0 = float(q2),
                                    x1= float(maximum),
                                    y0= 1.7,
                                    y1= 1.7,
                                    yref = 'paper',
                                    line=dict(
                                            color=c[0],
                                            width=3,
                                            )
                                    )
                shapes.append(shape)


                shape = go.layout.Shape(type='rect',
                                    x0 = float(q1),
                                    x1= float(q2),
                                    y0= 1.6,
                                    y1= 1.8,
                                    yref = 'paper',
                                    fillcolor = c[1],
                                    line=dict(
                                            color=c[0],
                                            width=3,
                                            )
                                    )
                shapes.append(shape)





        #get visualization ordering of features while visualizing them  (actual ordering of absolute SHAP values -> (feature name, visualization ordering))
        self.visualization_ordering = dict()
        count = 0   

        #visualize polygons for features with positive SHAP values
        first_pos_feature = True
        color_count = 0

        for elem in self.pos_x_coordinates:

            coords = elem[2]
            self.visualization_ordering[elem[3]] = (elem[0], count)
            count +=1
            
            if elem[0] in self.sample_impacting_features_names:
                
                if first_pos_feature:
                    
                    x =[coords[1]-0.001, coords[1]-0.001, coords[3]-0.005, coords[2], coords[3]-0.005, coords[1]-0.001] 
                    y =[0.7, 0.4, 0.4, 0.55, 0.7, 0.7] 
                else:
                    x=[coords[1]-0.006, coords[0]-0.001, coords[1]-0.006, coords[3]-0.005, coords[2], coords[3]-0.005, coords[1]-0.006]
                    y=[0.7, 0.55, 0.4, 0.4, 0.55, 0.7, 0.7]
                    
            else:

                if first_pos_feature:
                    x =[coords[1]-0.001, coords[1]-0.001, coords[3], coords[2], coords[3], coords[1]-0.001]
                    y =[0.65, 0.45, 0.45, 0.55, 0.65, 0.65]
            
                else: 
                    x=[coords[1]-0.001, coords[0]-0.001, coords[1]-0.001, coords[3], coords[2], coords[3], coords[1]-0.001]
                    y=[0.65, 0.55, 0.45, 0.45, 0.55, 0.65, 0.65]
            

            if color_count < len(self.red_colors): 
                color = self.red_colors[color_count]
            else:
                color = self.red_colors[-1]

            text_plus = ""
            if elem[0] in self.feature_value_range.keys():
                text_plus = "<br>Max:   " + str(self.feature_value_range[elem[0]][2]) + "<br>Average:   " + str(self.feature_value_range[elem[0]][1]) + "<br>Min:" + str(self.feature_value_range[elem[0]][0])
            
        
            trace = go.Scatter(x = x,
                               y = y,
                               fill = 'toself', 
                               fillcolor = 'hsv'+ str(color),
                               hoveron = 'fills',
                               line_color = '#FFFFFF' if highlighted_feature != elem[0] else "#000000",
                               name = elem[0] + " = " +  str(self.sample_feature_values[elem[0]]),
                               text= "<b>" + str(elem[0]) + " = " +  str(self.sample_feature_values[elem[0]]) + "<br>SHAP value: " +  "{:.3f}".format(elem[1]) + "</b><br>" + text_plus,
                               hoverinfo = 'text' ,
                               mode = 'lines',
                               line = dict(width = 1)
                               )
            traces.append(trace)
            first_pos_feature = False
            color_count+=1



        # visualize polygons for features with negative SHAP values
        first_neg_feature = True
        color_count = 0

        for elem in self.neg_x_coordinates:
            
            coords = elem[2]
            self.visualization_ordering[elem[3]] = (elem[0], count)
            count +=1

            if elem[0] in self.sample_impacting_features_names:
                if first_neg_feature:
                    x =[coords[1]+0.001, coords[1]+0.001, coords[3]+0.005, coords[2], coords[3]+0.005, coords[1]+0.001]
                    y =[0.7, 0.4, 0.4, 0.55, 0.7, 0.7]
                else:
                    x=[coords[1]+0.006, coords[0]+0.001, coords[1]+0.006, coords[3]+0.005, coords[2], coords[3]+0.005, coords[1]+0.006]
                    y=[0.7, 0.55, 0.4, 0.4, 0.55, 0.7, 0.7]
            else:
                if first_neg_feature:
                    x=[coords[1]+0.001, coords[1]+0.001, coords[3], coords[2], coords[3], coords[1]+0.001]
                    y =[0.65, 0.45, 0.45, 0.55, 0.65, 0.65]

                else:
                    x=[coords[1]+0.001, coords[0]+0.001, coords[1]+0.001, coords[3], coords[2], coords[3], coords[1]+0.001]
                    y=[0.65, 0.55, 0.45, 0.45, 0.55, 0.65, 0.65]
                   


            if color_count < len(self.blue_colors): 
                color = self.blue_colors[color_count]
            else:
                color = self.blue_colors[-1]

            text_plus = ""
            if elem[0] in self.feature_value_range.keys():
                text_plus = "<br>Max:   " + str(self.feature_value_range[elem[0]][2]) + "<br>Average:   " + str(self.feature_value_range[elem[0]][1]) + "<br>Min:   "  + str(self.feature_value_range[elem[0]][0])

            trace = go.Scatter(x=x,
                            y=y,
                            fill='toself', 
                            fillcolor= "hsv" + str(color),
                            hoveron = 'fills',
                            line_color= '#FFFFFF' if highlighted_feature != elem[0] else '#000000',
                            name = elem[0] + " = " +  str(self.sample_feature_values[elem[0]]),
                            text = "<b>" + str(elem[0])  + " = " + str(self.sample_feature_values[elem[0]]) + "<br>SHAP value: " +  "{:.3f}".format(elem[1]) + "</b><br>" + text_plus,
                            hoverinfo = 'text',
                            mode = 'lines',
                            line = dict(width = 1)
                            )
                
            traces.append(trace)
            first_neg_feature = False
            color_count+=1
        
            
                

        # draw max pos shap value over all samples
        trace = go.Scatter(x=[self.sample_prediction_prob, self.sample_prediction_prob - self.max_pos_shap_value[1]],
                            y=[0.85, 0.85],
                            hoveron = 'points',
                            line_color='#FF0D57',  
                            text= "SHAP value: " + str(self.max_pos_shap_value[1]) + "<br>feature: " + str(self.max_pos_shap_value[0]) 
                                    + "<br>sample id: " + str(self.max_pos_shap_value[2]),
                            hoverinfo = 'text',
                            mode = 'lines',
                            line = dict(
                                width = 4,
                            ),
                            showlegend = False,
        )
        traces.append(trace)
        


        # draw max neg shap value over all samples
        trace = go.Scatter(x=[self.sample_prediction_prob, self.sample_prediction_prob - self.max_neg_shap_value[1]],
                            y=[0.85, 0.85],
                            hoveron = 'points',
                            line_color='#1E88E5',
                            text=  "SHAP value: " + str(self.max_neg_shap_value[1]) + "<br>feature: " + str(self.max_neg_shap_value[0]) + "<br>sample id: " 
                                    + str(self.max_neg_shap_value[2]),
                            hoverinfo = 'text',
                            mode = 'lines',
                            line = dict(
                                width = 4,
                            ),
                            showlegend = False,
        )
        traces.append(trace)


        #draw shap value ranges
        if highlighted_feature != "":

            avg_pos, avg_neg, pos_count, neg_count, intervall = self.calculate_feature_value_range(highlighted_feature)
            
            if intervall is None:
                hovertext_pos = "average positive SHAP value for "+highlighted_feature+" = " + str(self.sample_feature_values[highlighted_feature]) + "<br>number of samples: " + str(pos_count),
                hovertext_neg = "average negative SHAP value for "+highlighted_feature+" = " + str(self.sample_feature_values[highlighted_feature]) + "<br>number of samples: "+str(neg_count),

            else: 
                hovertext_pos = "average positive SHAP value for "+highlighted_feature+" in " + str(intervall) + "<br>number of samples: " + str(pos_count),
                hovertext_neg = "average negative SHAP value for "+highlighted_feature+ " in " + str(intervall) + "<br>number of samples: "+str(neg_count),



            annotation = go.layout.Annotation(
                                    x = self.sample_prediction_prob - avg_pos - 0.01,
                                    y = 0.1,
                                    xanchor = "right",
                                    text = "|avg|↑ for " + highlighted_feature + " = " + str(self.sample_feature_values[highlighted_feature]) if intervall is None else "|avg|↑ for " + highlighted_feature + " in " + str(intervall),
                                    showarrow = False,
                                    font=go.layout.annotation.Font(
                                        color = '#FF0D57',
                                        size = 11
                                        ),           
                                    )
            annotations.append(annotation)


            annotation = go.layout.Annotation(
                                    x = self.sample_prediction_prob - avg_neg + 0.01,
                                    y = 0.1,
                                    xanchor = "left",
                                    text = "|avg|↓ for " + highlighted_feature + " = " + str(self.sample_feature_values[highlighted_feature]) if intervall is None else "|avg|↓ for " + highlighted_feature + " in " + str(intervall),
                                    showarrow = False,
                                    font=go.layout.annotation.Font(
                                        color = '#1E88E5',
                                        size = 11
                                        ),           
                                    )
            annotations.append(annotation)

            
            trace = go.Scatter( x=[self.sample_prediction_prob, self.sample_prediction_prob - avg_pos],
                                y=[0.1, 0.1],
                                hoveron = 'points',
                                line_color= "#FF0D57" ,
                                text=hovertext_pos,
                                hoverinfo = 'text',
                                mode = 'lines',
                                line = dict(
                                     width = 4,
                                ),
                                showlegend = False,
            )
            traces.append(trace)

            trace = go.Scatter( x=[self.sample_prediction_prob, self.sample_prediction_prob - avg_neg],
                                y=[0.1, 0.1],
                                hoveron = 'points',
                                line_color=  "#1E88E5",
                                text=hovertext_neg,
                                hoverinfo = 'text',
                                mode = 'lines',
                                line = dict(
                                     width = 4,
                                ),
                                showlegend = False,
            )
            traces.append(trace)


            



        #design layout of forceplot
        layout = go.Layout(
        hovermode='closest',
        plot_bgcolor = '#FFFFFF',
        autosize=False,
        width=1500,
        height = 400,
        legend = dict(
                x = 1.05,
                font= dict(
                    color = '#5E5E5E',
                    size = 10,
                    ),
        ),
        margin= dict(t = 250,
                     b = 0,
                     l = 0,
                     r = 0
        ),
        xaxis = go.layout.XAxis(
            position = 1,
            side = 'top',
            fixedrange = True,
            range = [self.min_x_coordinate - 0.05, self.max_x_coordinate + 0.05],
            showgrid = False,
            ticks = 'inside',
            zeroline = False,
            tickcolor = '#A6A6A6',
            tickfont = go.layout.xaxis.Tickfont(
                color = '#A6A6A6',
            ),
            tick0 = self.base_value,
            dtick = 0.1,
            nticks = 15,
            tickformat = '.3f',
            showline = True,
            linecolor = '#A6A6A6'
        ),
        yaxis = go.layout.YAxis(
            showgrid = False,
            showticklabels = False,
            fixedrange = True,
            range = [0, 1],
            zeroline = False,
            showline = False,
            linecolor = '#A6A6A6',
        ),
        
        annotations = annotations,
        shapes = shapes 
        )



        #create figure
        fig = go.Figure(data = traces, layout = layout)
        


        #create and add slider
        steps = []
        for i in range(len(self.visualization_ordering)):
            step = dict(
                method="update",
                args=[{"visible": [elem.visible for elem in fig.data]}],
                label = str(i+1)
                )
            for e in range (len(self.visualization_ordering)):
                step["args"][0]["visible"][self.visualization_ordering[e][1]] = False
            for e in range (i+1):
                step["args"][0]["visible"][self.visualization_ordering[e][1]] = True
            steps.append(step)
            
        
        fig.update_layout(
            sliders = [dict(
            active=len(self.sample_shap_values)-1,
            currentvalue={"prefix": "number of features shown: ",
                        "font": {
                            "size":15,
                            "color": '#5E5E5E',
                        }},
            tickcolor= '#5E5E5E',
            font={"color": '#5E5E5E',
                "size": 10 },
            ticklen= 5,
            len= 0.2,
            x= 0.8,
            y= 2.5,
            pad={"t": 0},
            bordercolor = '#5E5E5E',
            steps=steps,
        )]
        )

        fig.show()




    def draw_table(self):

        shapes = []
        traces = []
        annotations = []

        eval_ordering = dict()
        for elem in self.sample_shap_values:
            eval_ordering[elem[0]] = self.eval_impacting_features[elem[0]]
        self.eval_impacting_features = eval_ordering
        
        position = len(self.eval_impacting_features) - 0.5
        for item in self.eval_impacting_features.items():
           
            #draw NOT sole impact bars
            trace = go.Scatter(x=[0, item[1][0]],
                                y=[position, position],
                                hoveron = 'fills+points',
                                line_color="#70AD47",  
                                text= "impacting feature in {:.1f}".format(item[1][0]) + "% of the cases" ,
                                hoverinfo = 'text',
                                mode = 'lines',
                                line = dict(
                                    width = 16,
                                ),
                                name = item[0],
                                showlegend = False,
                                legendgroup = "impacting feature",
            )
            traces.append(trace)
            

            #draw sole impact bars
            trace = go.Scatter(x=[0, item[1][1]],
                                y=[position, position],
                                hoveron = 'fills+points',
                                line_color="#FF9300",  
                                text= "sole impacting feature in {:.1f}".format(item[1][1]) + "% of the cases" ,
                                hoverinfo = 'text',
                                mode = 'lines',
                                line = dict(
                                    width = 16,
                                ),
                                name = item[0],
                                showlegend = False,
                                legendgroup = "<i>sole</i> impacting feature",
            )
            traces.append(trace)
            


            #markers for feature names
            annotation = dict(text = item[0],
                            font= dict(color = '#5E5E5E',
                                    size = 12
                                    ),
                            showarrow = False,
                            x = 0.095,
                            y = position,
                            xref = 'paper',
                            xanchor = 'right',
                            yanchor = 'middle',
                            )
            annotations.append(annotation)

            position -=1


        
        #draw legend shape for NOT sole impacting features
        shape = go.layout.Shape(type='line',
                          x0= 0.77,
                          x1= 0.8,
                          y0= len(self.eval_impacting_features) - 0.5,
                          y1= len(self.eval_impacting_features) - 0.5,
                          xref = "paper",
                          line=dict(color="#70AD47",
                                    width=5,
                                    )
        )
        shapes.append(shape)

        #draw legend annotation for NOT sole impacting features
        annotation = go.layout.Annotation(  text = "cases in which feature<br>is impacting factor",
                                            font= dict(color = '#5E5E5E',
                                                    size = 10
                                                    ),
                                            showarrow = False,
                                            x = 0.81,
                                            y = len(self.eval_impacting_features) - 0.5,
                                            xref = 'paper',
                                            xanchor = 'left',
                                            yanchor = 'middle',

                          
        )
        annotations.append(annotation)


        #draw legend shape for sole impacting features
        shape = go.layout.Shape(type='line',
                          x0= 0.77,
                          x1= 0.8,
                          y0= len(self.eval_impacting_features) - 1.5 ,
                          y1= len(self.eval_impacting_features) - 1.5,
                          xref = "paper",
                          line=dict(color="#FF9300", 
                                    width=5,
                                    )
        )
        shapes.append(shape)

        #draw legend annotation for sole impacting features
        annotation = go.layout.Annotation(  text = "cases in which feature<br>is <i>sole</i> impacting factor",
                                            font= dict(color = '#5E5E5E',
                                                    size = 10
                                                    ),
                                            showarrow = False,
                                            x = 0.81,
                                            y = len(self.eval_impacting_features) - 1.5,
                                            xref = 'paper',
                                            xanchor = 'left',
                                            yanchor = 'middle',

                          
        )
        annotations.append(annotation)
            
            
    


        layout = go.Layout(
        title = dict(text = "Feature impact over all samples",
                     x = 0.1,
                     y = 0.92,
                     yref = "container",
                     font = dict(color = '#5E5E5E',
                                 size = 15,
            ),
        ),
        hovermode='closest',
        plot_bgcolor = '#FFFFFF',
        shapes = shapes,
        annotations = annotations,
        width = 1500,
        height = (len(self.eval_impacting_features) * 50) + 160,
        margin= dict(t = 80,
                     b = 80,

        ),
        xaxis = go.layout.XAxis(
            fixedrange = True,
            range = [0,100],
            showgrid = True,
            zeroline = False,
            gridcolor = "rgba(161, 161, 161, 0.3)",
            showline = True,
            showticklabels = True,
            domain = [0.1,0.7],
            linecolor = '#A6A6A6',
            tickcolor = '#A6A6A6',
            ticks = 'outside',
            ticksuffix  = "%",
            tickfont = go.layout.xaxis.Tickfont(
                color = '#5E5E5E',
            ),
            
        ),
        yaxis = go.layout.YAxis(
            fixedrange = True,
            range = [0, len(self.eval_impacting_features)],
            showgrid = True,
            gridcolor = "rgba(161, 161, 161, 0.3)",
            zeroline = False,
            showline = True,
            showticklabels = False,
            linecolor = '#A6A6A6',
            ticks = 'outside',
            tickcolor = '#A6A6A6',
            dtick = 1,
        ),
        )
       
        fig = go.Figure(data = traces, layout=layout)
        fig.show()






def select_mode(base_value, shap_values, classifier, features, X, encoded_X, sample_id, label, mode):

    le = My_Local_Explainer(base_value, shap_values, classifier, features, X, encoded_X, sample_id, label)

    if mode == "forceplot":
        le.draw_plot("", False)

    if mode == "output value box plots":
        le.draw_plot("", True)

    if mode == "impact evaluation":
        le.draw_plot("", False)
        le.draw_table()

    if mode == "shap value range":

        menu = Dropdown(options=[elem[0] for elem in le.sample_shap_values],
                    value=le.sample_shap_values[0][0],
                    description = "Feature:",
                    disabled=False,
                    layout=Layout(width='90%', 
                                  padding = "0px 635px 35px 415px",
                                  )
                   ) 

        interact(le.draw_plot, highlighted_feature = menu, boxplots = fixed(False))
    





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


def my_local_explainer(base_value, shap_values, classifier, features, X, label, decoder=dict()):
    
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

    textbox = BoundedIntText(value=0,
                             min=0,
                             max= len(X),
                             step=1,
                             description='Sample id:',
                             disabled=False,
                             layout=Layout(width='90%', 
                                  padding = "10px 640px 0px 420px",
                                  )
                            )

    menu = Dropdown(options=["forceplot", "output value box plots", "impact evaluation", "shap value range"],
                    value="forceplot",
                    description = "Mode:",
                    disabled=False,
                    layout=Layout(width='90%', 
                                  padding = "10px 640px 35px 420px",
                                  )
                   ) 
    
    
    interact(select_mode, base_value = fixed(base_value), shap_values = fixed(shap_values), classifier = fixed(classifier), features = fixed(features), X = fixed(decoded_X), 
            encoded_X = fixed(X), sample_id = textbox, label = fixed(label), mode = menu) 


