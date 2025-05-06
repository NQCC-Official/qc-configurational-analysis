# -*- coding: utf-8 -*-
"""

Save the results from the serverless VQE run

"""

import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_catalog import QiskitServerless, QiskitFunction

from qiskit_serverless import get, get_arguments, save_result

import qiskit_ibm_runtime as qiskit_ibm_runtime
import qiskit_ibm_catalog as qiskit_ibm_catalog
print(qiskit_ibm_runtime.__version__)
print(qiskit_ibm_catalog.__version__)
# instance = 'nqcc/nqcc---qat/digital-opt'

import time
import datetime
current_datetime = datetime.datetime.now()
timestr = time.strftime("%Y%m%d-%H%M%S")

# Add token and details
serverless = QiskitServerless(token="")
instance = ''
service = QiskitRuntimeService(channel="ibm_quantum", token="",
                              instance=instance)


# Print the job ids of your serverless runs
print(serverless.jobs())


#### Viewing and saving job results ####

# Enter job id you want to view
job = serverless.job("037b0582-714b-46f5-a70f-97cb1c209fe9") #
# # # # print(job.status())
print(job.logs())


result = job.result()

# Bug if you want to do multiple repeats in the session - this will give total QPU time in the session
# Works with one repeat at a time
QPU_ls = []
for j in range(int(result['repeats'])):
    # Enter session id from IBM quantum platform
    jobs = service.jobs(session_id='', limit=2000)

    total = 0
    its = 0
    for job in jobs:
        total += job.metrics()["usage"]["quantum_seconds"]
        its += 1
        # print(job.metrics())
    
    print('Total iterations:',its)
    
    print('Total QPU time:',total)
    timeQPU = total
    QPU_ls.append(timeQPU)

result['QPU_time'] = QPU_ls

try:
    for i in range(int(result['repeats'])): # 
        
        gen_path = result['device']+'_'+result['ansatz']+'_'+result['objective']+'_'+result['opt']+'_tol'+str(result['tol'])+'_'+result['prob']+'_l'+str(result['lambda'])+'_shts'+str(result['shots'])+f'_{timestr}'
        
        full_path_costevals ='RawResults/costevals_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_costevals, result['cost_history'][i], fmt='%.10f')
        
        full_path_params ='RawResults/params_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_params, result['param_history'][i], fmt='%.10f')
                    
        
        # !!! Might not save
        full_path_bits ='RawResults/bitstrings_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_bits, result['bitstrings'][i])
        # print('prob',prob)
        full_path_prob ='RawResults/prob_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_prob, result['probabilities'][i])
        
        # Time 
        path = 'RawResults/time_VQE_'+gen_path+f'_rep{i}.txt'
        with open(path, 'w') as file:
            file.write(str(result['user runtime'][i]))
            
        # Time stamps 
        path = 'RawResults/timeSTAMPS_VQE_'+gen_path+f'_rep{i}.txt'
        with open(path, 'w') as file:
            file.write(str(result['time_stamps']))
            
        # QPU Time 
        try:
            path = 'RawResults/QPUtime_VQE_'+gen_path+f'_rep{i}.txt'
            with open(path, 'w') as file:
                file.write(str(result['QPU_time'][i]))
        except:
            path = 'RawResults/QPUtime_VQE_'+gen_path+f'_rep{i}.txt'
            with open(path, 'w') as file:
                file.write('Save properly (session id)')
            
        # Save cost function time
        full_path_costfunTime = 'RawResults/costfunTime_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_costfunTime, result["cost_funct_time"][i])
        # Save iteration times
        full_path_iterationTime = 'RawResults/iterationTime_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_iterationTime, result["iter_time"][i])
        # Save profiler output
        path = 'RawResults/profileTime_VQE_'+gen_path+f'_rep{i}.txt'
        # Open file in write mode
        file = open(path, 'w')
        for i in range(len(result['profile_time'])):
            file.write(str(result['profile_time'][i]) + '\n')
        file.close()  # Close after all operations

except:
    
    print('Only a partial save!!!')
    
    for i in range(int(result['repeats'])): 
        
        gen_path = result['device']+'_'+result['ansatz']+'_'+result['objective']+'_'+result['opt']+'_tol'+str(result['tol'])+'_'+result['prob']+'_l'+str(result['lambda'])+'_shts'+str(result['shots'])
        
        full_path_costevals ='RawResults/costevals_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_costevals, result['cost_history'], fmt='%.10f')
        
        full_path_params ='RawResults/params_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_params, result['param_history'], fmt='%.10f')
    
        
        # !!! Might not save
        # full_path_bits ='1.RawResults/bitstrings_VQE_'+gen_path+f'_rep{i}.txt'
        # np.savetxt(full_path_bits, result['bitstrings'][i])
        # # print('prob',prob)
        # full_path_prob ='1.RawResults/prob_VQE_'+gen_path+f'_rep{i}.txt'
        # np.savetxt(full_path_prob, result['probabilities'][i])
        
        # Time 
        # path = '1.RawResults/time_VQE_'+gen_path+f'_rep{i}.txt'
        # with open(path, 'w') as file:
        #     file.write(str(result['user runtime'][i]))
            
        # Time stamps 
        # path = '1.RawResults/timeSTAMPS_VQE_'+gen_path+f'_rep{i}.txt'
        # with open(path, 'w') as file:
        #     file.write(str(result['time_stamps']))
            
        # QPU Time 
        try:
            path = 'RawResults/QPUtime_VQE_'+gen_path+f'_rep{i}.txt'
            with open(path, 'w') as file:
                file.write(str(result['QPU_time'][i]))
        except:
            path = 'RawResults/QPUtime_VQE_'+gen_path+f'_rep{i}.txt'
            with open(path, 'w') as file:
                file.write('Save properly (session id)')
            
        # Save cost function time
        # full_path_costfunTime = '1.RawResults/costfunTime_VQE_'+gen_path+f'_rep{i}.txt'
        # np.savetxt(full_path_costfunTime, result["cost_funct_time"][i])
        # Save iteration times
        full_path_iterationTime = 'RawResults/iterationTime_VQE_'+gen_path+f'_rep{i}.txt'
        np.savetxt(full_path_iterationTime, result["iter_time"])
        # Save profiler output
        # path = '1.RawResults/profileTime_VQE_'+gen_path+f'_rep{i}.txt'
        # # Open file in write mode
        # file = open(path, 'w')
        # for i in range(len(result['profile_time'])):
        #     file.write(str(result['profile_time'][i]) + '\n')
        # file.close()  # Close after all operations

