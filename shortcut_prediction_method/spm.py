import pandas as pd
import numpy as np
import math
from dateutil.parser import parse
from scipy.spatial.distance import pdist, squareform
from datetime import datetime, timedelta

class ShortcutPrediction(object):
    def __init__(self, data, lam):
        self.data = data
        self.lam = lam

    def top_and_bottom_values(self):
        values = np.array(self.data.values)
        mins = np.array([min(self.data)] * len(self.data))
        positive_series = pd.Series(values - mins, index = self.data.index)
        index = []
        top_and_bottom_values = []
        candidate_time = None
        candidate_value = None
        for i in range(len(self.data)):
            if i == 0:
                index.append(positive_series.index[i])
                top_and_bottom_values.append(positive_series[i])
            else:
               if candidate_value == None:
                   if positive_series[i] >= ((2 + self.lam) / (2 - self.lam)) * top_and_bottom_values[0] or positive_series[i] <= ((2 - self.lam) / (2 + self.lam)) * top_and_bottom_values[0]:
                       candidate_time = positive_series.index[i]
                       candidate_value = positive_series[i]
               else:
                   if candidate_value > top_and_bottom_values[-1]:
                       if positive_series[i] > candidate_value:
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
                       elif positive_series[i] <= ((2 - self.lam) / (2 + self.lam)) * candidate_value:
                           index.append(candidate_time)
                           top_and_bottom_values.append(candidate_value)
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
                   elif candidate_value < top_and_bottom_values[-1]:
                       if positive_series[i] < candidate_value:
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
                       elif positive_series[i] >= ((2 + self.lam) / (2 - self.lam)) * candidate_value:
                           index.append(candidate_time)
                           top_and_bottom_values.append(candidate_value)
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
        top_and_bottom_values = pd.Series(np.array(top_and_bottom_values) + mins[:len(top_and_bottom_values)], index = index)
        return top_and_bottom_values

    def top_and_bottom_ids(self):
        values = np.array(self.data.values)
        mins = np.array([min(self.data)] * len(self.data))
        positive_series = pd.Series(values - mins, index = self.data.index)
        index = []
        top_and_bottom_values = []
        top_and_bottom_ids = []
        candidate_time = None
        candidate_value = None
        for i in range(len(self.data)):
            if i == 0:
                index.append(positive_series.index[i])
                top_and_bottom_values.append(positive_series[i])
                top_and_bottom_ids.append(i)
            else:
               if candidate_value == None:
                   if positive_series[i] >= ((2 + self.lam) / (2 - self.lam)) * top_and_bottom_values[0] or positive_series[i] <= ((2 - self.lam) / (2 + self.lam)) * top_and_bottom_values[0]:
                       candidate_time = positive_series.index[i]
                       candidate_value = positive_series[i]
               else:
                   if candidate_value > top_and_bottom_values[-1]:
                       if positive_series[i] > candidate_value:
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
                       elif positive_series[i] <= ((2 - self.lam) / (2 + self.lam)) * candidate_value:
                           index.append(candidate_time)
                           top_and_bottom_values.append(candidate_value)
                           top_and_bottom_ids.append(i)
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
                   elif candidate_value < top_and_bottom_values[-1]:
                       if positive_series[i] < candidate_value:
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
                       elif positive_series[i] >= ((2 + self.lam) / (2 - self.lam)) * candidate_value:
                           index.append(candidate_time)
                           top_and_bottom_values.append(candidate_value)
                           top_and_bottom_ids.append(i)
                           candidate_time = positive_series.index[i]
                           candidate_value = positive_series[i]
        top_and_bottom_ids = pd.Series(np.array(top_and_bottom_ids), index = index)
        return top_and_bottom_ids

    def embedding_dimension(self, delay_time):
        deltas = (self.top_and_bottom_values() - self.top_and_bottom_values().shift(1)).dropna()
        data_frame = pd.DataFrame(deltas, index = deltas.index)
        previous_deimension_data_frame = data_frame
        next_deimension_data_frame = pd.merge(data_frame, data_frame.shift(delay_time), right_index = True, left_index = True)
        for i in range(math.floor(len(deltas) / delay_time) - 2):
            previous_deimension_data_frame = next_deimension_data_frame
            previous_dist = pd.DataFrame(squareform(pdist(previous_deimension_data_frame, metric='euclidean')), columns = data_frame.index, index = data_frame.index)
            next_deimension_data_frame = pd.merge(next_deimension_data_frame, data_frame.shift((i + 2) * delay_time), right_index = True, left_index = True)
            next_dist = pd.DataFrame(squareform(pdist(next_deimension_data_frame, metric='euclidean')), columns = data_frame.index, index = data_frame.index)
            fnn_count = 0
            for j in range(len(previous_dist)):
                if j > (i + 2) * delay_time:
                    shaped_previous_dist = previous_dist[data_frame.index[j]].dropna()
                    shaped_previous_dist = shaped_previous_dist.drop(data_frame.index[j])
                    min_id = shaped_previous_dist.argmin()
                    DL = math.sqrt((pow(next_dist[data_frame.index[j]][min_id], 2) - pow(shaped_previous_dist[min_id], 2)) / pow(shaped_previous_dist[min_id], 2))
                    if DL >= 15:
                        fnn_count = fnn_count + 1
            if fnn_count == 0:
                return i + 2

    def predict(self, delay_time):
        embedding_dimension = self.embedding_dimension()
        top_and_bottom_values = self.top_and_bottom_values()
        top_and_bottom_ids = self.top_and_bottom_ids()
        deltas = (top_and_bottom_values - top_and_bottom_values.shift(1)).dropna()
        steps = (top_and_bottom_ids - top_and_bottom_ids.shift(1)).dropna()
        pred = {'value': top_and_bottom_values.values[-1] + float(self.local_linear_approximation(deltas, delay_time, embedding_dimension)),
                'step': top_and_bottom_ids.values[-1] + round(self.local_linear_approximation(steps, delay_time, embedding_dimension)) - (len(self.data) - 1)}
        return pred

    def local_linear_approximation(self, data, delay_time, embedding_dimension):
        neighbors = []
        for i in range(embedding_dimension):
            if i == 0:
                index = [x for x in range(len(data))]
                data_frame = pd.DataFrame(data.values, index = index)
            else:
                data_frame = pd.merge(data_frame, data_frame.shift(i * delay_time), right_index = True, left_index = True)
        dist = pd.DataFrame(squareform(pdist(data_frame, metric = 'euclidean')), columns = index, index = index)
        dist = dist[dist.index[-1]].dropna()
        sorted_dist = dist.sort_values()
        for j in [x for x in range(embedding_dimension + 2)]:
            if len(sorted_dist) - 1 >= j:
                if sorted_dist.index[j] != dist.index[-1] and len(neighbors) < embedding_dimension + 1:
                    neighbors.append(sorted_dist.index[j])
        predicted_values = []
        candidates = data.shift(-1)
        for k in range(len(neighbors)):
            predicted_values.append(candidates[neighbors[k]])
        return np.average(np.array(predicted_values))
