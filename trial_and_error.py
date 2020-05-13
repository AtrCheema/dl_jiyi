
from optimization import objective_func


inputs = {'1': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '2': ['pcp_mm','tide_cm', 'wat_temp_c',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '3': ['pcp_mm','tide_cm', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '4': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '5': ['pcp_mm','tide_cm',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '6': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa'],
          '7': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'rel_hum'],
          '8': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'mslp_hpa', 'rel_hum'],
          '9': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '10': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '11': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '12': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa'],
          '13': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps', 'rel_hum'],
          '14': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'mslp_hpa', 'rel_hum'],
          '15': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '16': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'wind_speed_mps', 'air_p_hpa', 'mslp_hpa', 'rel_hum'],
          '17': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'wind_dir_deg', 'wind_speed_mps', 'air_p_hpa', 'mslp_hpa'],
          '18': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'wind_speed_mps'],
          '19': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'wind_dir_deg', 'rel_hum'],
          '20': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_temp_c', 'mslp_hpa', 'rel_hum'],
          '21': ['pcp_mm','tide_cm', 'wat_temp_c', 'sal_psu',
               'air_p_hpa', 'mslp_hpa', 'rel_hum']
          }


for k, in_features in inputs.items():

    error, prediction_error = objective_func(in_features, BatchSize=32,
                       lookback=int(8),
                       lr=float(1e-6),
                       lstm_units=int(128),
                       act_f='relu')