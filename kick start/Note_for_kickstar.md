1. 实现等比数列求和，如下图，其中j，k都很大，结果需要对1e9+7取模

   ![1584429128556](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1584429128556.png)

   ![1584428940041](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1584428940041.png)

   ![1584428967486](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1584428967486.png)

   ```c++
   #include <iostream>
   #include <vector>
   #include <cmath>
   using namespace std;
   
   unsigned long long MOD = 1e9+7;
   unsigned long long fastPow(unsigned long long x, unsigned long long n){
       if(x == 1)
           return x;
       if(n == 0)
           return 1;
       unsigned long long ans = fastPow(x, n/2);
       if(n % 2 == 0)
           return (ans * ans) % MOD;//注意%优先级高于*
       else
           return ((ans * ans) % MOD * x) % MOD;//注意%优先级高于*
   }
              
   //i*(i^k-1)/(i-1)
   dengbiSUM[i] = (i* (fastPow(i,k) - 1) % MOD * (fastPow(i-1, MOD-2) % MOD)) % MOD;
   ```

2. 啊

3. 