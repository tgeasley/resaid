import pandas as pd
import numpy as np
import sys
import numpy_financial as npf
from tqdm import tqdm

class well_econ:

    def __init__(self):
        #Constants
        self.STAT_FILE = sys.stdout
        self.OIL_COL = "OIL"
        self.GAS_COL = 'GAS'
        self.WATER_COL = 'WATER'

        #Settable
        self._flowstreams = None
        self._header_data = None

        self._flowstream_uwi_col = None
        self._flowstream_t_index = None
        self._header_uwi_col = None

        self._royalty = None

        self._opc_t = None
        self._opc_oil = None
        self._opc_gas = None
        self._opc_water = None

        self._scale_capex = False
        self._scale_column = None
        self._capex_val = None

        self._atx = None
        self._sev_gas = None
        self._sev_oil = None

        self._oil_pri = None
        self._gas_pri = None
        
        self._discount_rate = None

        self._breakeven_phase = None


        #Get only
        self._indicators = None

        

    @property
    def flowstreams(self):
        return self._flowstreams


    @flowstreams.setter
    def flowstreams(self,value):
        self._flowstreams = value

    @property
    def flowstream_uwi_col(self):
        return self._flowstream_uwi_col


    @flowstream_uwi_col.setter
    def flowstream_uwi_col(self,value):
        self._flowstream_uwi_col = value

    @property
    def flowstream_t_index(self):
        return self._flowstream_t_index


    @flowstream_t_index.setter
    def flowstream_t_index(self,value):
        self._flowstream_t_index = value

    @property
    def header_uwi_col(self):
        return self._header_uwi_col


    @header_uwi_col.setter
    def header_uwi_col(self,value):
        self._header_uwi_col = value

    @property
    def header_data(self):
        return self._header_data


    @header_data.setter
    def header_data(self,value):
        self._header_data = value

    @property
    def royalty(self):
        return self._royalty


    @royalty.setter
    def royalty(self,value):
        self._royalty = value

    @property
    def opc_t(self):
        return self._opc_t


    @opc_t.setter
    def opc_t(self,value):
        self._opc_t = value

    @property
    def opc_oil(self):
        return self._opc_oil


    @opc_oil.setter
    def opc_oil(self,value):
        self._opc_oil = value

    @property
    def opc_gas(self):
        return self._opc_gas


    @opc_gas.setter
    def opc_gas(self,value):
        self._opc_gas = value

    @property
    def opc_water(self):
        return self._opc_water


    @opc_water.setter
    def opc_water(self,value):
        self._opc_water = value

    @property
    def scale_capex(self):
        return self._scale_capex


    @scale_capex.setter
    def scale_capex(self,value):
        self._scale_capex = value

    @property
    def scale_column(self):
        return self._scale_column


    @scale_column.setter
    def scale_column(self,value):
        self._scale_column = value

    @property
    def capex_val(self):
        return self._capex_val


    @capex_val.setter
    def capex_val(self,value):
        self._capex_val = value

    @property
    def atx(self):
        return self._atx


    @atx.setter
    def atx(self,value):
        self._atx = value

    @property
    def sev_gas(self):
        return self._sev_gas


    @sev_gas.setter
    def sev_gas(self,value):
        self._sev_gas = value

    @property
    def sev_oil(self):
        return self._sev_oil


    @sev_oil.setter
    def sev_oil(self,value):
        self._sev_oil = value

    @property
    def oil_pri(self):
        return self._oil_pri


    @oil_pri.setter
    def oil_pri(self,value):
        self._oil_pri = value

    @property
    def gas_pri(self):
        return self._gas_pri


    @gas_pri.setter
    def gas_pri(self,value):
        self._gas_pri = value

    @property
    def discount_rate(self):
        return self._discount_rate


    @discount_rate.setter
    def discount_rate(self,value):
        self._discount_rate = value

    @property
    def indicators(self):
        return self._indicators

    @property
    def breakeven_phase(self):
        return self._breakeven_phase

    @breakeven_phase.setter
    def breakeven_phase(self,value):
        self._breakeven_phase = value

   

    def generate_oil_price(self,times):
        oil_price = []
        if isinstance(self._oil_pri, list):
            if len(self._oil_pri) >= len(times):
                oil_price = self._oil_pri[0:len(times)]
            else:
                last_pri = self._oil_pri[-1]
                num_to_add = len(times)-len(self._oil_pri)
                add_list = [last_pri for i in range(num_to_add)]
                oil_price = self._oil_pri
                oil_price.extend(add_list)
        else:
            oil_price = [self._oil_pri for i in range(len(times))]

        return np.array(oil_price)

    def generate_gas_price(self,times):
        gas_price = []
        if isinstance(self._gas_pri, list):
            if len(self._gas_pri) >= len(times):
                gas_price = self._gas_pri[0:len(times)]
            else:
                last_pri = self._gas_pri[-1]
                num_to_add = len(times)-len(self._gas_pri)
                add_list = [last_pri for i in range(num_to_add)]
                gas_price = self._gas_pri
                gas_price.extend(add_list)
        else:
            gas_price = [self._gas_pri for i in range(len(times))]

        return np.array(gas_price)

    def generate_capex(self,times,well):
        l_capex = np.zeros(times)
        #print(well)
        #print(self._scale_column, self._scale_capex, file=self.STAT_FILE, flush=True)
        if self._scale_capex:
            #print(self._scale_column, file=self.STAT_FILE, flush=True)
            #print(self._header_data[self._header_uwi_col])
            scale_val = self._header_data[self._header_data[self._header_uwi_col]==well].iloc[0][self._scale_column]
            #print(scale_val)
            #print(scale_val, file=self.STAT_FILE, flush=True)
            #scale_val = scale_val[self._scale_column]
            l_capex[0] = self.capex_val*scale_val
        else:
            l_capex[0] = self._capex_val

        return l_capex

    def zero_below(self,df:pd.DataFrame,i_max:int, cols:list):
        for col in cols:
            df[col] = np.where(
                df['T_INDEX'] <= i_max,
                df[col],
                0
            )

        return df

    def generate_indicators(self):

        ind_dict = {
            'UWI':[],
            'EURO':[],
            'EURG':[],
            'EURW':[],
            'REVENUE':[],
            'ROYALTY':[],
            'OPEX':[],
            'TAXES':[],
            'CAPEX':[],
            'FCF':[],
            'DCF':[],
            'IRR':[],
            'BREAKEVEN':[],
            'BREAKEVEN_PHASE':[]
        }

        unique_wells = self._flowstreams[self._flowstream_uwi_col].unique()

        for w in tqdm(unique_wells):
            l_flow = self._flowstreams[self._flowstreams[self._flowstream_uwi_col]==w].reset_index()
            t_series = np.array(range(len(l_flow)))

            l_flow['oil_price'] = self.generate_oil_price(t_series)
            l_flow['gas_price'] = self.generate_gas_price(t_series)

            l_flow['revenue'] = (
                l_flow[self.OIL_COL]*l_flow['oil_price']+
                l_flow[self.GAS_COL]*l_flow['gas_price']
            )

            l_flow['royalty'] = l_flow['revenue']*self._royalty

            l_flow['expense'] = (
                self._opc_t+
                self._opc_gas*l_flow[self.GAS_COL]+
                self._opc_oil*l_flow[self.OIL_COL]+
                self._opc_water*l_flow[self.WATER_COL]
            )

            l_flow['taxes'] = (
                self._atx*l_flow['revenue']+
                self._sev_gas*l_flow[self.GAS_COL]*l_flow['gas_price']+
                self._sev_oil*l_flow[self.OIL_COL]*l_flow['oil_price']
            )*(1-self._royalty)

            l_flow['capex'] = self.generate_capex(len(l_flow),w)

            l_flow['cf'] = (
                l_flow['revenue'] -
                l_flow['royalty'] -
                l_flow['expense'] -
                l_flow['taxes'] -
                l_flow['capex']
            )


            l_flow['dcf'] = (l_flow['cf'].to_numpy() / (1+self._discount_rate)**np.arange(0, len(l_flow['cf'].to_numpy())))
            

            try:
                cf_idx = np.argwhere(l_flow['dcf'].to_numpy()>0)
            except:
                cf_idx=[]

            if len(cf_idx) > 0:
                last_cf = np.max(np.argwhere(l_flow['dcf'].to_numpy()>0))
            else:
                last_cf = 0

            zero_cols = [
                self.OIL_COL,
                self.GAS_COL,
                self.WATER_COL,
                'revenue',
                'royalty',
                'expense',
                'taxes',
                'capex',
                'cf',
                'dcf'
            ]

            l_flow = self.zero_below(l_flow,last_cf,zero_cols)
            
            dc_rev = (l_flow['revenue'].to_numpy() / (1+self._discount_rate)**np.arange(0, len(l_flow['revenue'].to_numpy())))

            if self._breakeven_phase is None:
                if np.sum(l_flow[self.OIL_COL]) > 0:
                    if np.sum(l_flow[self.GAS_COL])/np.sum(l_flow[self.OIL_COL])> 3.2:
                        be_major = 'GAS'
                        break_even =(np.sum(dc_rev)- np.sum(l_flow['dcf']))/np.sum(l_flow[self.GAS_COL])
                    else:    
                        be_major = 'OIL'
                        break_even =(np.sum(dc_rev)- np.sum(l_flow['dcf']))/np.sum(l_flow[self.OIL_COL])
                else:
                    be_major = 'GAS'
                    break_even =(np.sum(dc_rev)- np.sum(l_flow['dcf']))/np.sum(l_flow[self.GAS_COL])
            else:
                if self._breakeven_phase == "GAS":
                    be_major = 'GAS'
                    break_even =(np.sum(dc_rev)- np.sum(l_flow['dcf']))/np.sum(l_flow[self.GAS_COL])
                else:
                    be_major = 'OIL'
                    break_even =(np.sum(dc_rev)- np.sum(l_flow['dcf']))/np.sum(l_flow[self.OIL_COL])

            ind_dict['UWI'].append(w)
            ind_dict['EURO'].append(np.sum(l_flow[self.OIL_COL]))
            ind_dict['EURG'].append(np.sum(l_flow[self.GAS_COL]))
            ind_dict['EURW'].append(np.sum(l_flow[self.WATER_COL]))
            ind_dict['REVENUE'].append(np.sum(l_flow['revenue']))
            ind_dict['ROYALTY'].append(np.sum(l_flow['royalty']))
            ind_dict['OPEX'].append(np.sum(l_flow['expense']))
            ind_dict['TAXES'].append(np.sum(l_flow['taxes']))
            ind_dict['CAPEX'].append(np.sum(l_flow['capex']))
            ind_dict['FCF'].append(np.sum(l_flow['cf']))
            ind_dict['DCF'].append(np.sum(l_flow['dcf']))
            ind_dict['BREAKEVEN'].append(break_even)
            ind_dict['BREAKEVEN_PHASE'].append(be_major)

            #if np.sum(cf_array) > 0:
            try:
                ind_dict['IRR'].append(np.power(1+npf.irr(l_flow['cf'].to_numpy()),12)-1)
            except:
                ind_dict['IRR'].append(0)
            #else:
            #    ind_dict['IRR'].append(0)


        l_flow.to_csv('outputs/test_cf.csv')
        self._indicators = pd.DataFrame(ind_dict)