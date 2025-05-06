# -*- coding: utf-8 -*-
"""

Upload VQE job with serverless

"""

from qiskit_ibm_catalog import QiskitServerless, QiskitFunction


import qiskit_ibm_runtime as qiskit_ibm_runtime
import qiskit_ibm_catalog as qiskit_ibm_catalog
print(qiskit_ibm_runtime.__version__)
print(qiskit_ibm_catalog.__version__)
# instance = 'nqcc/nqcc---qat/digital-opt'

import time
import datetime
current_datetime = datetime.datetime.now()
timestr = time.strftime("%Y%m%d-%H%M%S")



####  Upload Job ####
# Add token
serverless = QiskitServerless(token="")

VQE = QiskitFunction(
    title="VQE_serverless",
    entrypoint="VQE_QPU.py",
    working_dir="./source_files/",
)

serverless.upload(VQE)

VQE_remote_serverless = serverless.load("VQE_serverless")

job = VQE_remote_serverless.run()

print(serverless.jobs())


