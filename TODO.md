# STACK: State Covariance with a Kalman filter

## Schedule  

`target date` 20250317
1. Obtain a small set of IGb20 SINEX files and extract 3 stations for testing using pytrf's sinex.py. Specifically use the keep_sta method. 

`target date` 20250323
2. Create a Stack.cpp. 
3. Create library function XsnxToXstack.cpp. Incorporate M-PAGES SINEX class to read measurements and covariances from X.snx. 

`target date` 20250331
4. Incorporate Kalman class in Stack.cpp.  
5. Implement GLOBK-like frame stabilization scheme.  

