
from optimization import objective_func


inputs = {'1': ['pcp_mm','tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm','tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm','tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '1': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '1': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '1': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '1': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '1': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '1': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm', 'tide_cm', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum']
          }


for k, in_features in inputs.items():

    error, prediction_error = objective_func(in_features, BatchSize=32,
                       lookback=int(8),
                       lr=float(1e-6),
                       lstm_units=int(128),
                       act_f='relu')