from re import S
import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from dateutil.relativedelta import *
import time

import warnings
from tqdm import tqdm
warnings.simplefilter("ignore")

class decline_curve:

    def __init__(self):
        #Constants
        self.DAY_NORM = 30.4
        self.GAS_CUTOFF = 3.2 #GOR for classifying well as gas or oil, MSCF/STB
        self.STAT_FILE = sys.stdout
        self.MINOR_TAIL = 6 #Number of months from tail to use for minor phase ratios
        self.SET_LENGTH = 5280 #Length to normalize horizontals to
        self.D_MIN = .08/12 #Minimum monthly decline rate
        self.DEBUG_ON = False
        #Settable 
        self.verbose = True
        self.FILTER_BONFP = .5 #Normally set to .5
        self.DEFAULT_DI  = .8/12
        self.DEFAULT_B = .5
        self.V_DCA_FAILURES = 0
        self._min_h_b = .99
        self._max_h_b = 2
        
        self._backup_decline = False
        self._dataframe = None
        self._date_col = None
        self._phase_col = None
        self._length_col = None
        self._uid_col = None
        self._dayson_col = None
        self._oil_col = None
        self._gas_col = None
        self._water_col = None

        #Get only variables
        self._normalized_dataframe = pd.DataFrame()
        self._params_dataframe = None
        self._flowstream_dataframe = None
        self._typecurve = None

    @property
    def dataframe(self):
        return self._dataframe


    @dataframe.setter
    def dataframe(self,value):
        self._dataframe = value

    @property
    def date_col(self):
        return self._date_col


    @date_col.setter
    def date_col(self,value):
        self._date_col = value

    @property
    def phase_col(self):
        return self._phase_col


    @phase_col.setter
    def phase_col(self,value):
        self._phase_col = value

    @property
    def length_col(self):
        return self._length_col


    @length_col.setter
    def length_col(self,value):
        self._length_col = value

    @property
    def uid_col(self):
        return self._uid_col


    @uid_col.setter
    def uid_col(self,value):
        self._uid_col = value

    @property
    def dayson_col(self):
        return self._dayson_col


    @dayson_col.setter
    def dayson_col(self,value):
        self._dayson_col = value

    @property
    def oil_col(self):
        return self._oil_col


    @oil_col.setter
    def oil_col(self,value):
        self._oil_col = value

    @property
    def gas_col(self):
        return self._gas_col


    @gas_col.setter
    def gas_col(self,value):
        self._gas_col = value

    @property
    def water_col(self):
        return self._water_col


    @water_col.setter
    def water_col(self,value):
        self._water_col = value


    @property
    def backup_decline(self):
        return self._backup_decline


    @backup_decline.setter
    def backup_decline(self,value):
        self._backup_decline = value

    @property
    def min_h_b(self):
        return self._min_h_b


    @min_h_b.setter
    def min_h_b(self,value):
        self._min_h_b= value


    @property
    def max_h_b(self):
        return self._max_h_b


    @max_h_b.setter
    def max_h_b(self,value):
        self._max_h_b= value

    @property
    def params_dataframe(self):
        return self._params_dataframe

    @property
    def flowstream_dataframe(self):
        return self._flowstream_dataframe

    @property
    def typecurve(self):
        return self._typecurve

    def month_diff(self, a, b):
        return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)

    def infill_production(self):
        """
        An error was found where gaps in the historical production would be infilled
        with the wrong P_DATE
        """

    def generate_t_index(self):
        #print(self._date_col, file=self.STAT_FILE, flush=True)
        self._dataframe[self._date_col] = pd.to_datetime(self._dataframe[self._date_col])
        min_by_well = self._dataframe[[self._uid_col,self._date_col]].groupby(by=[self._uid_col]).min().reset_index()
        min_by_well = min_by_well.rename(columns={self._date_col:'MIN_DATE'})
        #print(min_by_well)
        
        self._dataframe = self._dataframe.merge(
            min_by_well, 
            left_on = self._uid_col,
            right_on = self._uid_col,
            suffixes=(None,'_MIN')
        )

        self._dataframe['T_INDEX'] = self.month_diff(
            self._dataframe[self._date_col],
            self._dataframe['MIN_DATE']
        )

        #return 0

    def assign_major(self):
        l_cum = self._normalized_dataframe[['UID','NORMALIZED_OIL','NORMALIZED_GAS']].groupby(by=['UID']).sum().reset_index()
        l_cum['MAJOR'] = np.where(
            l_cum["NORMALIZED_OIL"] >0,
            np.where(
                l_cum["NORMALIZED_GAS"]/l_cum['NORMALIZED_OIL']>self.GAS_CUTOFF,
                'GAS',
                'OIL'
            ),
            "GAS"
        )

        self._normalized_dataframe = self._normalized_dataframe.merge(
            l_cum,
            left_on = "UID",
            right_on = "UID",
            suffixes=(None,'_right')
        )

    def normalize_production(self):

        self._normalized_dataframe['UID'] = self._dataframe[self._uid_col]
        self._normalized_dataframe['T_INDEX'] = self._dataframe['T_INDEX']

        if self._length_col == None:
            self._normalized_dataframe['LENGTH_NORM'] = 1
        else:
            self._dataframe[self._length_col] = self._dataframe[self._length_col].fillna(0)

            self._normalized_dataframe['LENGTH_NORM'] = np.where(
                self._dataframe[self._length_col] > 1,
                self._dataframe[self._length_col],
                1
            )

        self._normalized_dataframe['HOLE_DIRECTION'] = np.where(
            self._normalized_dataframe['LENGTH_NORM']> 1,
            "H",
            "V"
        )

        if self._length_col == None:
            self._normalized_dataframe['LENGTH_SET'] = 1
        else:
            self._normalized_dataframe['LENGTH_SET'] = np.where(
                self._dataframe[self._length_col] > 1,
                self.SET_LENGTH,
                1
            )

        if self._dayson_col == None:
            self._normalized_dataframe['DAYSON'] = 30.4
        else:
            self._dataframe[self._dayson_col] = self._dataframe[self._dayson_col].fillna(30.4)

            self._normalized_dataframe['DAYSON'] = np.where(
                self._dataframe[self._dayson_col] > 0,
                self._dataframe[self._dayson_col],
                0
            )

        self._normalized_dataframe['NORMALIZED_OIL'] = (
            self._dataframe[self._oil_col]*
            self.DAY_NORM*
            self._normalized_dataframe['LENGTH_SET'] /
            (self._normalized_dataframe['LENGTH_NORM'] * self._normalized_dataframe['DAYSON'])
        )

        self._normalized_dataframe['NORMALIZED_GAS'] = (
            self._dataframe[self._gas_col]*
            self.DAY_NORM*
            self._normalized_dataframe['LENGTH_SET'] /
            (self._normalized_dataframe['LENGTH_NORM'] * self._normalized_dataframe['DAYSON'])
        )

        self._normalized_dataframe['NORMALIZED_WATER'] = (
            self._dataframe[self._water_col]*
            self.DAY_NORM*
            self._normalized_dataframe['LENGTH_SET'] /
            (self._normalized_dataframe['LENGTH_NORM'] * self._normalized_dataframe['DAYSON'])
        )

        
        if self._phase_col == None:
            self.assign_major()
        else:
            self._normalized_dataframe['MAJOR'] = self._dataframe[self._phase_col]
        

        self._normalized_dataframe = self._normalized_dataframe[[
            'UID',
            'LENGTH_NORM',
            "HOLE_DIRECTION",
            'MAJOR',
            'T_INDEX',
            'NORMALIZED_OIL',
            'NORMALIZED_GAS',
            'NORMALIZED_WATER'
        ]]

        #if self.DEBUG_ON:
        self._normalized_dataframe.to_csv('outputs/norm_test.csv')
    
    def outlier_detection(self, input_x, input_y):

        
        filtered_x = []
        filtered_y = []
    
        ln_input_y= np.log(input_y)

        if len([i for i in ln_input_y if i > 0]) > 0:
            
            regression = sm.formula.ols("data ~ x", data=dict(data=ln_input_y, x=input_x)).fit()
            try:
                test = regression.outlier_test()
                
                for index, row in test.iterrows():
                    if row['bonf(p)']> self.FILTER_BONFP:
                        filtered_x.append(input_x[index])
                        filtered_y.append(input_y[index])
            except:
                if self.verbose:
                    print('Error in outlier detection.')
                filtered_x = input_x
                filtered_y = input_y

        return filtered_x, filtered_y

    def arps_decline(self,x,qi,di,b,t0):
        if qi > 0:
            problemX = t0-1/(b*di)
            #print(di,self.D_MIN,b,qi)
            if di < self.D_MIN:
                di = self.D_MIN
                tlim = -1
            else:
                qlim = qi*(self.D_MIN/di)**(1/b)
                #print(qlim)
                try:
                    tlim = int(((qi/qlim)**(b)-1)/(b*di)+t0)
                    #q_at_lim = (qi)/(1+b*(di)*(int(tlim)-t0))**(1/b)
                except:
                    print(qi,qlim,di,b)
            #problemX = t0+1
            #print(tlim)
            q_x = np.where(
                x>problemX,
                np.where(x<tlim,
                    (qi)/(1+b*(di)*(x-t0))**(1/b),
                    qlim*np.exp(-self.D_MIN*(x-tlim))
                ),
                0
            )
            #print(q_x)
            #qi = (qi)/(1+b*(ai)*(x))**(1/b)
        else:
            q_x = [0.0 for _ in x]
        return q_x
    
    def handle_dca_error(self,s,x_vals,y_vals):
        if s["MAJOR"] == 'OIL':
            #print(sum_df)
            minor_ratio = np.sum(s['NORMALIZED_GAS'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_OIL'][-self.MINOR_TAIL:])
            water_ratio = np.sum(s['NORMALIZED_WATER'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_OIL'][-self.MINOR_TAIL:])
        else:
            minor_ratio = np.sum(s['NORMALIZED_OIL'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_GAS'][-self.MINOR_TAIL:])
            water_ratio = np.sum(s['NORMALIZED_WATER'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_GAS'][-self.MINOR_TAIL:])
        i = -1
        while i > -len(x_vals):
            if y_vals[i]>0:
                break
            else:
                i -= 1
        s['qi']=y_vals[i]
        s['di']=self.DEFAULT_DI
        s['b']=self.DEFAULT_B
        s['t0']=x_vals[i]
        s['q0']=y_vals[0] #Probably will need revision, high chance first value is zero
        s['minor_ratio']=minor_ratio
        s['water_ratio']=water_ratio

        return s

    def dca_params(self,s):

        x_vals = s['T_INDEX']

        if s['MAJOR'] == 'OIL':
            y_vals = s['NORMALIZED_OIL']
        else:
            y_vals = s['NORMALIZED_GAS']

        if len(x_vals) > 3:
            z = np.array(y_vals)
            a = argrelextrema(z, np.greater)
            if len(a[0]) > 0:
                indexMax = a[-1][-1]
                indexMin = a[-1][0]
                t0Max = x_vals[indexMax]
                t0Min = x_vals[indexMin]
            else:
                indexMax = 0
                indexMin = 0
                t0Max = x_vals[indexMax]
                t0Min = x_vals[indexMin]
            

            filtered_x = np.array(x_vals[indexMin:])
            filtered_y = np.array(y_vals[indexMin:])

            zero_filter = np.array([y > 0 for y in filtered_y])
            filtered_x = filtered_x[zero_filter]
            filtered_y = filtered_y[zero_filter]
            
            outliered_x, outliered_y = self.outlier_detection(filtered_x,filtered_y)

            

            if len(outliered_x) > 3:
                if t0Min == t0Max:
                    t0Max = t0Max + 1

                di_int = np.log(outliered_y[0]/outliered_y[-1])/(outliered_x[-1]-outliered_x[0])
                q_max = np.max(outliered_y)
                q_min = np.min(outliered_y)

                if s['HOLE_DIRECTION'] == 'H':
                    bMin = self._min_h_b
                    bMax = self._max_h_b
                else:
                    bMin = .01
                    bMax = .99

                if di_int < 0:
                    di_int = np.log(q_max/q_min)/(outliered_x[outliered_y.index(q_min)]-outliered_x[outliered_y.index(q_max)])
                
                if di_int < 0:
                    if q_max == outliered_y[-1]:
                        di_int = .1
                    else:
                        di_int = np.log(q_max/outliered_y[-1])/(outliered_x[-1]-outliered_x[outliered_y.index(q_max)])
                
                
                weight_range = list(range(1,len(outliered_x)+1))
                weight_range = weight_range[::-1]
                
                try:
                    popt, pcov = curve_fit(self.arps_decline, outliered_x, outliered_y,
                        p0=[q_max, di_int,(bMin+bMax)/2,t0Min], 
                        bounds=([q_min,di_int/2,bMin, t0Min], [q_max*1.1,2*di_int,bMax,t0Max]),
                        sigma = weight_range, absolute_sigma = True)
                    
                    

                    if s["MAJOR"] == 'OIL':
                        #print(sum_df)
                        minor_ratio = np.sum(s['NORMALIZED_GAS'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_OIL'][-self.MINOR_TAIL:])
                        water_ratio = np.sum(s['NORMALIZED_WATER'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_OIL'][-self.MINOR_TAIL:])
                    else:
                        minor_ratio = np.sum(s['NORMALIZED_OIL'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_GAS'][-self.MINOR_TAIL:])
                        water_ratio = np.sum(s['NORMALIZED_WATER'][-self.MINOR_TAIL:])/np.sum(s['NORMALIZED_GAS'][-self.MINOR_TAIL:])

                    s['qi']=popt[0]
                    s['di']=popt[1]
                    s['b']=popt[2]
                    s['t0']=popt[3]
                    s['q0']=y_vals[0] #Probably will need revision, high chance first value is zero
                    s['minor_ratio']=minor_ratio
                    s['water_ratio']=water_ratio
                except:
                    self.V_DCA_FAILURES += 1
                    if self.verbose:
                        print('DCA Error: '+str(s['UID']), file=self.STAT_FILE, flush=True)

                    if self._backup_decline:
                        #local_df = self._normalized_dataframe.loc[self._normalized_dataframe['UID'] == w]
                        #sum_df = local_df.tail(self.MINOR_TAIL).sum()
                        return self.handle_dca_error(s,x_vals, y_vals)
            else:
                self.V_DCA_FAILURES += 1
                if self.verbose:
                    print('Base x: {}  Filtered x: {}  Outliered x: {}'.format(len(x_vals),len(filtered_x),len(outliered_x)))
                    print('Insufficent data after filtering, well: '+str(s['UID']), file=self.STAT_FILE, flush=True)
                if self._backup_decline:
                    return self.handle_dca_error(s,x_vals, y_vals)

        else :
            self.V_DCA_FAILURES += 1
            if self.verbose:
                print('Insufficent data before filtering, well: '+str(s['UID']), file=self.STAT_FILE, flush=True)
            if self._backup_decline:
                return self.handle_dca_error(s,x_vals, y_vals)

        return s

    def vect_generate_params(self):
        self.V_DCA_FAILURES = 0
        l_start = time.time()

        imploded_df = self._normalized_dataframe[[
            'UID',
            'MAJOR',
            'HOLE_DIRECTION',
            'LENGTH_NORM',
            'T_INDEX',
            'NORMALIZED_OIL',
            'NORMALIZED_GAS',
            'NORMALIZED_WATER'
        ]].groupby(
            ['UID',
            'MAJOR',
            'HOLE_DIRECTION',
            'LENGTH_NORM']
        ).agg({
            'T_INDEX': lambda x: x.tolist(),
            'NORMALIZED_OIL': lambda x: x.tolist(),
            'NORMALIZED_GAS': lambda x: x.tolist(),
            'NORMALIZED_WATER': lambda x: x.tolist()
        }).reset_index()

        imploded_df = imploded_df.apply(self.dca_params, axis=1)

        imploded_df = imploded_df[[
            'UID',
            'MAJOR',
            'LENGTH_NORM',
            'q0',
            'qi',
            'di',
            'b',
            't0',
            'minor_ratio',
            'water_ratio',
        ]].rename(columns={
            'MAJOR':'major',
            'LENGTH_NORM':'h_length'
        })

        r_df:pd.DataFrame = pd.DataFrame([])

        for major in ['OIL','GAS']:
            l_df = imploded_df[imploded_df['major']==major]

            if len(l_df)>0:

                q3, q2, q1 = np.percentile(l_df['minor_ratio'], [75,50 ,25])
                high_cutoff = 1.5*(q3-q1)+q3
                l_df['minor_ratio'] = np.where(
                    l_df['minor_ratio']>high_cutoff,
                    q2,
                    l_df['minor_ratio']
                )

                q3, q2, q1 = np.percentile(l_df['water_ratio'], [75,50 ,25])
                high_cutoff = 1.5*(q3-q1)+q3
                l_df['water_ratio'] = np.where(
                    l_df['water_ratio']>high_cutoff,
                    q2,
                    l_df['water_ratio']
                )

                if r_df.empty:
                    r_df = l_df
                else:
                    r_df = pd.concat([r_df,l_df])

        imploded_df = r_df

        print('Total DCA Failures: '+str(self.V_DCA_FAILURES), file=self.STAT_FILE, flush=True)
        print(f'Total wells analyzed: {len(imploded_df)}', file=self.STAT_FILE, flush=True)
        print('Failure rate: {:.2%}'.format(self.V_DCA_FAILURES/len(imploded_df)), file=self.STAT_FILE, flush=True)
        l_duration = time.time() - l_start
        print("Vectorized DCA generation: {:.2f} seconds".format(l_duration), file=self.STAT_FILE, flush=True)

        self._params_dataframe = imploded_df


    def run_DCA(self, _verbose=True):
        self.verbose = _verbose
        if self.verbose:
            print('Generating time index.', file=self.STAT_FILE, flush=True)
            
        
        self.generate_t_index()

        if self.verbose:
            print('Normalizing production.', file=self.STAT_FILE, flush=True)

        self.normalize_production()

        if self.verbose:
            print('Generating decline parameters.', file=self.STAT_FILE, flush=True)
        #self.generate_params()
        
        self.vect_generate_params()

    def add_months(self, start_date, delta_period):
        end_date = start_date + pd.DateOffset(months=delta_period)
        return end_date

    def generate_flowstream(self, num_months=1200, denormalize=False, actual_dates=False, _verbose=False):
        self.verbose = _verbose

        if self._params_dataframe == None:
            self.run_DCA(_verbose=_verbose)

        t_range = np.array(range(1,num_months))

        flow_dict = {
            'UID':[],
            'MAJOR':[],
            'T_INDEX':[],
            'OIL':[],
            'GAS':[],
            'WATER':[]
        }

        

        for index, row in self._params_dataframe.iterrows():
            if denormalize and row['h_length']>1:
                denormalization_scalar = row['h_length']/self.SET_LENGTH
            else:
                denormalization_scalar = 1

            dca = np.array(self.arps_decline(t_range,row.qi,row.di,row.b,row.t0))*denormalization_scalar
            if np.sum(dca) > 0:
                flow_dict['UID'].append(row['UID'])
                flow_dict['MAJOR'].append(row['major'])
                flow_dict['T_INDEX'].append(t_range)
                if row['major'] == "OIL":
                    flow_dict['OIL'].append(dca)
                    if np.isnan(row['minor_ratio']):
                        flow_dict['GAS'].append(dca*0)
                    else:
                        flow_dict['GAS'].append(dca*row['minor_ratio'])
                else:
                    flow_dict['GAS'].append(dca)
                    if np.isnan(row['minor_ratio']):
                        flow_dict['OIL'].append(dca*0)
                    else:
                        flow_dict['OIL'].append(dca*row['minor_ratio'])
                if np.isnan(row['water_ratio']):
                    flow_dict['WATER'].append(dca*0)
                else:
                    flow_dict['WATER'].append(dca*row['water_ratio'])


        self._flowstream_dataframe = pd.DataFrame(flow_dict)
        #print(self._flowstream_dataframe.columns)
        self._flowstream_dataframe = self._flowstream_dataframe.set_index(['UID','MAJOR']).apply(pd.Series.explode).reset_index()
        self._flowstream_dataframe = self._flowstream_dataframe.set_index(['UID', 'T_INDEX'])

        self._flowstream_dataframe['OIL'] = pd.to_numeric(
            self._flowstream_dataframe['OIL']
        )

        self._flowstream_dataframe['GAS'] = pd.to_numeric(
            self._flowstream_dataframe['GAS']
        )

        self._flowstream_dataframe['WATER'] = pd.to_numeric(
            self._flowstream_dataframe['WATER']
        )

        if denormalize:
            actual_df = self._dataframe[[self._uid_col,'T_INDEX',self._oil_col,self._gas_col,self._water_col]]
            actual_df = actual_df.rename(columns={
                self._uid_col:'UID',
                self._oil_col:'OIL',
                self._gas_col:"GAS",
                self._water_col:"WATER"
            })
        else:
            actual_df = self._normalized_dataframe[[
                'UID',
                'T_INDEX',
                'NORMALIZED_OIL',
                'NORMALIZED_GAS',
                'NORMALIZED_WATER'
            ]]
            actual_df = actual_df.rename(columns={
                'NORMALIZED_OIL':'OIL',
                'NORMALIZED_GAS':"GAS",
                'NORMALIZED_WATER':'WATER'
            })

        if actual_dates:
            actual_df['P_DATE'] = self._dataframe[self._date_col]
            self._flowstream_dataframe['P_DATE'] = None
            
        # Added to ensure UID and T_INDEX are unique
    
        actual_df = actual_df.set_index(['UID', 'T_INDEX'])

        self._flowstream_dataframe.update(actual_df)
        self._flowstream_dataframe = self._flowstream_dataframe.reset_index()


        if actual_dates:
            
            # Updated code using the fact that T_INDEX is always referenced to MIN_DATE

            self._flowstream_dataframe['P_DATE'] = pd.to_datetime(self._flowstream_dataframe['P_DATE'])

            min_df = self._dataframe[[self._uid_col,self._date_col]].groupby(by=[self._uid_col]).min().reset_index()
            #min_df = self._flowstream_dataframe.groupby(['UID']).min().reset_index()
            min_df = min_df.rename(columns={self._uid_col:"UID",self._date_col:"MIN_DATE"})
            min_df = min_df[min_df['MIN_DATE'].notnull()]

            self._flowstream_dataframe = self._flowstream_dataframe.merge(min_df, left_on='UID', right_on='UID')

            self._flowstream_dataframe = self._flowstream_dataframe.replace([np.inf, -np.inf], np.nan)

            self._flowstream_dataframe['P_DATE'] = np.where(
                self._flowstream_dataframe['P_DATE'].isnull(),
                self._flowstream_dataframe.apply(lambda row: self.add_months(row["MIN_DATE"], row["T_INDEX"]), axis = 1),
                self._flowstream_dataframe['P_DATE']
            )

            self._flowstream_dataframe = self._flowstream_dataframe.drop(['MIN_DATE'],axis=1)

            # Orginal code encountered issues when date skips in historical were present

            #self._flowstream_dataframe['P_DATE'] = pd.to_datetime(self._flowstream_dataframe['P_DATE'])
            
            #cum_count = self._flowstream_dataframe[self._flowstream_dataframe['P_DATE'].isnull()].groupby(['UID']).cumcount().rename('OFFSET_INDEX')
            #cum_count = cum_count+1
            #self._flowstream_dataframe = self._flowstream_dataframe.merge(cum_count,how='left', left_index=True, right_index=True)
            
            #max_df = self._flowstream_dataframe.groupby(['UID']).max().reset_index()
            #max_df = max_df[['UID','P_DATE']].rename(columns={'P_DATE':'MAX_DATE'})
            #max_df = max_df[max_df['MAX_DATE'].notnull()]
            
            #self._flowstream_dataframe = self._flowstream_dataframe.merge(max_df, left_on='UID', right_on='UID')
            
            #self._flowstream_dataframe = self._flowstream_dataframe.replace([np.inf, -np.inf], np.nan)
            
            #self._flowstream_dataframe['OFFSET_INDEX'] = self._flowstream_dataframe['OFFSET_INDEX'].fillna(0)

            #self._flowstream_dataframe['P_DATE'] = np.where(
            #    self._flowstream_dataframe['P_DATE'].isnull(),
            #    self._flowstream_dataframe.apply(lambda row: self.add_months(row["MAX_DATE"], row["OFFSET_INDEX"]), axis = 1),
            #    self._flowstream_dataframe['P_DATE']
            #)

            #self._flowstream_dataframe = self._flowstream_dataframe.drop(['OFFSET_INDEX','MAX_DATE'],axis=1)

    def generate_typecurve(self, num_months=1200, denormalize=False, prob_levels=[.1,.5,.9], _verbose=False):
        if self._flowstream_dataframe == None:
            self.generate_flowstream(num_months=num_months,denormalize=denormalize, _verbose=_verbose)

        return_df = self._flowstream_dataframe.reset_index()
        #print(return_df.head())
        #return_df = self._flowstream_dataframe[['T_INDEX','OIL','GAS','WATER']]
        if self.DEBUG_ON:
            return_df.to_csv('outputs/test_quantiles.csv')
        return_df = self._flowstream_dataframe.groupby(['T_INDEX']).quantile(prob_levels).reset_index()
        #print(return_df.head())
        return_df = return_df.pivot(
            index=['T_INDEX'],
            columns='level_1',
            values=['OIL','GAS','WATER']
        )
        #oil_df.columns = ['P10 Oil, bbl/(km-month)','P50 Oil, bbl/(km-month)','P90 Oil, bbl/(km-month)']
        #oil_df = oil_df.rename(columns={'T_INDEX':'Months Online'})

        self._typecurve = return_df