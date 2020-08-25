### 78. 子集（中等）

---

1. 题目描述

   给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

   说明：解集不能包含重复的子集。

   ```
   示例:
   输入: nums = [1,2,3]
   输出:
   [
     [3],
     [1],
     [2],
     [1,2,3],
     [1,3],
     [2,3],
     [1,2],
     []
   ]
   ```

2. 简单实现

   ```c++
   class Solution {
   public:
       vector<vector<int>> ans;
       vector<int> temp;
       
       void dfs(vector<int>& nums, int i)
       {
           if(i == nums.size())
               ans.push_back(temp);
           else {
               temp.push_back(nums[i]);
               dfs(nums,i+1);//放进去
               temp.pop_back();
               
               dfs(nums,i+1);//不放
           }
       }
       vector<vector<int>> subsets(vector<int>& nums) {
           if(nums.size() <= 0)
               return ans;
           else
               dfs(nums,0);
           return ans;
       }
   };
   ```

### 118. 杨辉三角（简单）

---

1. 题目描述

   给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。

   ![img](https://upload.wikimedia.org/wikipedia/commons/0/0d/PascalTriangleAnimated2.gif)

   在杨辉三角中，每个数是它左上方和右上方的数的和。

   ```
   示例:
   输入: 5
   输出:
   [
        [1],
       [1,1],
      [1,2,1],
     [1,3,3,1],
    [1,4,6,4,1]
   ]
   ```

2. 简单实现

   ```c++
   class Solution {
   public:
       vector<vector<int>> generate(int numRows) {
           vector<vector<int>> ans;
           if(numRows == 0)
               return ans;
           ans.push_back({1});
           if(numRows == 1)
               return ans;
           ans.push_back({1,1});
           if(numRows == 2)
               return ans;
           for(int idx = 3; idx <= numRows; idx++){
               vector<int> temp = vector<int>(idx, 1);
               for(int i = 1; i < idx - 1; i++){
                   temp[i] = ans[idx-2][i-1]+ans[idx-2][i];
               }
               ans.push_back(temp);
           }
           return ans;
       }
   };
   ```

### 105. 从前序与中序遍历序列构造二叉树（中等）

---

1. 题目描述

   根据一棵树的前序遍历与中序遍历构造二叉树。

   注意: 你可以假设树中没有重复的元素。

       例如，给出
       前序遍历 preorder = [3,9,20,15,7]
       中序遍历 inorder = [9,3,15,20,7]
       返回如下的二叉树：
          3
         / \
         9  20
           /  \
          15   7

2. 简单实现

   ```c++
   class Solution {
   public:
       unordered_map<int, int> m;
       void init(vector<int>& v){
           for(int i = 0; i < v.size(); i++) m[v[i]] = i;
       }
       
       TreeNode* build(vector<int>& inorder, int l1, int r1, vector<int>& preorder, int l2, int r2){
           if(l1 == r1) return new TreeNode(inorder[l1]);//只有一个节点，作为根节点返回
           TreeNode* root = new TreeNode(preorder[l2]);//前序遍历的第一个节点值为根节点值
           //找到根节点在中序遍历中的位置，其左方为左子树中序遍历结果，右方为右子树中序遍历结果
           int idx = m[preorder[l2]];
           if(idx != l1)//有左子树
               root->left = build(inorder, l1, idx-1, preorder, l2+1, l2+idx-l1);
           if(idx != r1)//有右子树
               root->right = build(inorder, idx+1, r1, preorder, r2-(r1-idx)+1, r2); 
           return root;
       }
       TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
           int len = inorder.size();
           if(len == 0) return NULL;
           if(len == 1) return new TreeNode(inorder[0]);//只有一个节点，作为根节点返回
           init(inorder);
           return build(inorder, 0, len-1, preorder, 0, len-1);
       }
   };
   ```

### 112. Fizz Buzz（简单）

---

1. 题目描述

   写一个程序，输出从 1 到 n 数字的字符串表示。

   1. 如果 n 是3的倍数，输出“Fizz”；
   2. 如果 n 是5的倍数，输出“Buzz”；
   3. 如果 n 同时是3和5的倍数，输出 “FizzBuzz”。

   ```
   示例：
   n = 15,
   返回:
   [
       "1",
       "2",
       "Fizz",
       "4",
       "Buzz",
       "Fizz",
       "7",
       "8",
       "Fizz",
       "Buzz",
       "11",
       "Fizz",
       "13",
       "14",
       "FizzBuzz"
   ]
   ```

2. 简单实现

   ```c++
   class Solution {
   public:
       vector<string> fizzBuzz(int n) {
           vector<string> ans(n);
           for(int i = 1; i <= n; i++){
               if(i % 15 == 0)
                   ans[i-1] = "FizzBuzz";
               else if(i % 3 == 0)
                   ans[i-1] = "Fizz";
               else if(i % 5 == 0)
                   ans[i-1] = "Buzz";
               else
                   ans[i-1] = to_string(i);
           }
           return ans;
       }
   };
   ```

### 457. 环形数组循环（中等）

----

1. 题目描述

   给定一个含有正整数和负整数的环形数组 nums。 如果某个索引中的数 k 为正数，则向前移动 k 个索引。相反，如果是负数 (-k)，则向后移动 k 个索引。因为数组是环形的，所以可以假设最后一个元素的下一个元素是第一个元素，而第一个元素的前一个元素是最后一个元素。

   确定 nums 中是否存在循环（或周期）。循环必须在相同的索引处开始和结束并且循环长度 > 1。此外，一个循环中的所有运动都必须沿着同一方向进行。换句话说，一个循环中不能同时包括向前的运动和向后的运动。

   ```
   示例 1：
   输入：[2,-1,1,2,2]
   输出：true
   解释：存在循环，按索引 0 -> 2 -> 3 -> 0 。循环长度为 3 。
   
   示例 2：
   输入：[-1,2]
   输出：false
   解释：按索引 1 -> 1 -> 1 ... 的运动无法构成循环，因为循环的长度为 1 。根据定义，循环的长度必须大于 1 。
   
   示例 3:
   输入：[-2,1,-1,-2,-2]
   输出：false
   解释：按索引 1 -> 2 -> 1 -> ... 的运动无法构成循环，因为按索引 1 -> 2 的运动是向前的运动，而按索引 2 -> 1 的运动是向后的运动。一个循环中的所有运动都必须沿着同一方向进行。
   ```


   提示：

   - -1000 ≤ nums[i] ≤ 1000
   - nums[i] ≠ 0
   - 0 ≤ nums.length ≤ 5000


   进阶：你能写出时间时间复杂度为 O(n) 和额外空间复杂度为 O(1) 的算法吗？

2. 正确解法

   ![1589427107794](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1589427107794.png)

   ```c++
   class Solution {
   public:
       bool circularArrayLoop(vector<int>& nums) {
           vector<int> visit(nums.size(),0);
           int color = 1;
           for(int i=0;i<nums.size();i++) {
               if(visit[i] == 0) {//没探查过
                   int j = i;
                   while(visit[j] == 0 && nums[j]*nums[i]>0) {//没人探查过且同向移动
                       visit[j] = color;
                       j = j + nums[j] + nums.size();
                       j = j%nums.size();
                   }
                   if(visit[j] == color && (j+nums[j]+nums.size())%nums.size() != j)//找到环且长度大于1
                       return true;
               }
               color++;
           }
           return false;
       }
   };
   ```

3. 最优解法

   快慢指针

   ```c++
   //循环问题：用快慢指针解决
   //计算 next 位置，对于会超出数组的长度的正数，我们可以通过对n取余，但是对于负数，若这个负数远大于n的话，取余之前只加上一个n，可能是不够的，所以正确的方法是应该先对n取余，再加上n。为了同时把正数的情况也包含进来，最终我们的处理方法是先对n取余，再加上n，再对n取余，这样不管正数还是负数，大小如何，都可以成功的旋转跳跃了。
   int getnext(int* nums,int i,int numsSize) {
       return (((nums[i] + i) % numsSize) + numsSize) % numsSize;
   }
   
   bool circularArrayLoop(int* nums, int numsSize){
       for(int i = 0;i < numsSize;i++) {
           if(nums[i] == 0) {//如果遇到已经访问过，被标记的地方
               continue;//我们也可以不用 visited 数组，直接在 nums 中标记，由于题目中说了 nums 中不会有0，所以可以把访问过的位置标记为0
           }
           int slow = i;//对于每个i位置，慢指针指向i
           int fast = getnext(nums,i,numsSize);//快指针指向下一个位置，这里调用子函数来计算下一个位置
           int val = nums[i];
           while(val * nums[fast] > 0 && val * nums[getnext(nums,fast,numsSize)] > 0) {//慢指针指向的数要和快指针指向的数正负相同，这个不难理解。并且慢指针指向的数还要跟快指针的下一个位置上的数符号相同
               if(slow == fast) {
                   if(slow == getnext(nums,slow,numsSize)) {//直到当快慢指针相遇的时候，就是环出现的时候，但是这里有个坑，即便快慢指针相遇了，也不同立马返回 true，因为题目中说了了环的长度必须大于1，所以我们要用慢指针指向的数和慢指针下一个位置上的数比较，若相同，则说明环的长度为1，此时并不返回 false，而且 break 掉 while 循环。因为这只能说以i位置开始的链表无符合要求的环而已，后面可能还会出现符合要求的环。但是若二者不相同的话，则已经找到了符合要求的环，直接返回 true。
                       break;
                   }
                   return true;
               }
               //若快慢指针还不相同的，则分别更新，慢指针走一步，快指针走两步
               slow = getnext(nums,slow,numsSize);
               fast = getnext(nums,getnext(nums,fast,numsSize),numsSize);
           }
           //当 while 循环退出后，我们需要标记已经走过的结点，从而提高运算效率，方法就是将慢指针重置为i，再用一个 while 循环，条件是 nums[i] 和 慢指针指的数正负相同
           slow = i;
           while(val * nums[slow] > 0) {
               int next = getnext(nums,slow,numsSize);//然后计算下一个位置，并且 nums[slow] 标记为0，并且慢指针移动到 next 位置。
               nums[slow] = 0;
               slow = next;
           }
       }
       //最终 for 循环退出后，返回 false 即可
       return false;
   }
   ```

### 39. 组合总和（中等）

---

1. 题目描述

   给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。 candidates 中的数字可以无限制重复被选取。

   说明：

   - 所有数字（包括 target）都是正整数。
   - 解集不能包含重复的组合。 

   ```
   示例 1:
   输入: candidates = [2,3,6,7], target = 7,
   所求解集为:
   [
     [7],
     [2,2,3]
   ]
   
   示例 2:
   输入: candidates = [2,3,5], target = 8,
   所求解集为:
   [
     [2,2,2,2],
     [2,3,3],
     [3,5]
   ]
   ```

2. 简单实现——DFS

   ```c++
   class Solution {
   public:
       vector<vector<int>> ans;
       void dfs(vector<int>& candidates, int i, int target, vector<int>& temp, int sum){
           if(sum == target){
               ans.push_back(temp);
               return;
           }
           if(sum > target) return;
           if(i == candidates.size() || candidates[i] > target) return;
           dfs(candidates, i+1, target, temp, sum);//不加candidates[i]
           vector<int> back = temp;
           int bak = sum;
           while(sum <= target){//一个一个加candidates[i]
               sum += candidates[i];
               temp.push_back(candidates[i]);
               dfs(candidates, i+1, target, temp, sum);
           }
           temp = back;
           sum = bak;
       }
       vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
           sort(candidates.begin(), candidates.end(), greater<int>());//从大到小排序，这样可以先排除一些
           vector<int> temp;
           dfs(candidates, 0, target, temp, 0);
           return ans;
       }
   };
   ```

### 141. 环形链表（简单）

---

1. 题目描述

   给定一个链表，判断链表中是否有环。

   为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。

   ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

   ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test2.png)

   ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist_test3.png)

    ```
   示例 1：
   输入：head = [3,2,0,-4], pos = 1
   输出：true
   解释：链表中有一个环，其尾部连接到第二个节点。
   
   示例 2：
   输入：head = [1,2], pos = 0
   输出：true
   解释：链表中有一个环，其尾部连接到第一个节点。
   
   示例 3：
   输入：head = [1], pos = -1
   输出：false
   解释：链表中没有环。
    ```


   进阶：你能用 O(1)（即，常量）内存解决此问题吗？

2. 简单实现

   ```c++
   class Solution {
   public:
       bool hasCycle(ListNode *head) {
           if(!head || !head->next)
               return false;
           ListNode *slow = head;
           ListNode *fast = head->next;
           while(fast != slow){
               if(fast->next && fast->next->next) fast = fast->next->next;
               else return false;
               slow = slow->next;
           }
           if(!fast) return false;
           else return true;
       }
   };
   ```

### 318. 最大单词长度乘积（中等）

---

1. 题目描述

   给定一个字符串数组 words，找到 length(word[i]) * length(word[j]) 的最大值，并且这两个单词不含有公共字母。你可以认为每个单词只包含小写字母。如果不存在这样的两个单词，返回 0。

   ```
   示例 1:
   输入: ["abcw","baz","foo","bar","xtfn","abcdef"]
   输出: 16 
   解释: 这两个单词为 "abcw", "xtfn"。
   
   示例 2:
   输入: ["a","ab","abc","d","cd","bcd","abcd"]
   输出: 4 
   解释: 这两个单词为 "ab", "cd"。
   
   示例 3:
   输入: ["a","aa","aaa","aaaa"]
   输出: 0 
   解释: 不存在这样的两个单词。
   ```

2. 简单实现

   位运算

   ```c++
   class Solution {
   public:
       static bool cmp(string& a, string& b){
           return a.size() > b.size();
       }
       int maxProduct(vector<string>& words) {
           int size = words.size();
           if(size <= 1) return 0;
           sort(words.begin(), words.end(), cmp);//按长度从达到小排序
           vector<int> v(size, 0);//用二进制数记录各个单词包含各个字母的情况
           for(int i = 0; i < size; i++){
               for(int j = 0; j < words[i].size(); j++){
                   v[i] |= 1 << (words[i][j] - 'a');
               }
           }
           int ans = 0;
           for(int i = 0; i < size - 1; i++){
               int lena = words[i].size();
               for(int j = i+1; j < size; j++){
                   if(lena * words[j].size() <= ans)//长度没有ans长，不用看了
                       break;
                   if((v[i] & v[j]) == 0)
                       ans = lena * words[j].size();
               }
           }
           return ans;
       }
   };
   ```

### 739. 每日温度（中等）

---

1. 题目描述

   根据每日 气温 列表，请重新生成一个列表，对应位置的输出是需要再等待多久温度才会升高超过该日的天数。如果之后都不会升高，请在该位置用 0 来代替。

   例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

   提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。

2. 简单实现——单调栈

   ```c++
   class Solution {
   public:
       vector<int> dailyTemperatures(vector<int>& T) {
           int size = T.size();
           vector<int> ans(size);
           ans[size-1] = 0;
           stack<int> s;//单调递减栈
           s.push(size-1);
           int idx = size - 2;
           while(idx >= 0){//从后往前遍历
               while(!s.empty() && T[s.top()] <= T[idx])
                   s.pop();
               if(s.empty())
                   ans[idx] = 0;
               else
                   ans[idx] = s.top() - idx;
               s.push(idx);
               idx--;
           }
           return ans;
       }
   };
   ```

### 378. 有序矩阵中第K小的元素（中等）

---

1. 题目描述

   给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第k小的元素。
   请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。

    ```
   示例:
   matrix = [
      [ 1,  5,  9],
      [10, 11, 13],
      [12, 13, 15]
   ],
   k = 8,
   返回 13。
    ```


   提示：你可以假设 k 的值永远是有效的, 1 ≤ k ≤ n2 。

2. 简单实现

   ```c++
   class Solution {
   public:
       int kthSmallest(vector<vector<int>>& matrix, int k) {
           int n = min(int(matrix.size()), k);
           priority_queue<int> q;
           for(int i = 0; i < n; i++)
               for(int j = 0; j < n; j++){
                   if(q.size() < k)
                       q.push(matrix[i][j]);
                   else if(matrix[i][j] < q.top()){
                       q.pop();
                       q.push(matrix[i][j]);
                   }
               }
           return q.top();
       }
   };
   ```

3. 二分法

   ![1589769027440](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1589769027440.png)

   ![1589769059062](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1589769059062.png)

### 401. 二进制手表（简单）

---

1. 题目描述

   二进制手表顶部有 4 个 LED 代表小时（0-11），底部的 6 个 LED 代表分钟（0-59）。

   每个 LED 代表一个 0 或 1，最低位在右侧。

   ![img](https://upload.wikimedia.org/wikipedia/commons/8/8b/Binary_clock_samui_moon.jpg)

   例如，上面的二进制手表读取 “3:25”。
   给定一个非负整数 n 代表当前 LED 亮着的数量，返回所有可能的时间。

   ```
   案例:
   输入: n = 1
   返回: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
   ```


   注意事项:

   - 输出的顺序没有要求。
   - 小时不会以零开头，比如 “01:00” 是不允许的，应为 “1:00”。
   - 分钟必须由两位数组成，可能会以零开头，比如 “10:2” 是无效的，应为 “10:02”。

2. 简单实现

   ```c++
   class Solution {
   public:
       vector<string> ans;
       string getTime(int h, int m){
           string ans = "";
           ans += to_string(h) + ':';
           if(m < 10)
               ans += '0';
           ans += to_string(m);
           return ans;
       }
       void dfs(int cur, int num, unordered_set<int>& visited){
           if(num == 0){
               ans.push_back(getTime(cur >> 6, cur & 0x3F));
               return;
           }
           int n = 1;
           for(int i = 0; i < 10; i++){
               int tmp = cur | (1 << i);
               int h = tmp >> 6;
               int m = tmp & 0x3F;
               if(h > 11 || m > 59) continue;
               if(visited.find(tmp) == visited.end()){
                   visited.insert(tmp);
                   dfs(tmp, num-1, visited);
               }
           }
       }
       vector<string> readBinaryWatch(int num) {
           unordered_set<int> visited;
           dfs(0, num, visited);
           return ans;
       }
   };
   ```

### 1170. 比较字符串最小字母出现频次（简单）

---

1. 题目描述

   我们来定义一个函数 f(s)，其中传入参数 s 是一个非空字符串；该函数的功能是统计 s  中（按字典序比较）最小字母的出现频次。

   例如，若 s = "dcce"，那么 f(s) = 2，因为最小的字母是 "c"，它出现了 2 次。

   现在，给你两个字符串数组待查表 queries 和词汇表 words，请你返回一个整数数组 answer 作为答案，其中每个 answer[i] 是满足 f(queries[i]) < f(W) 的词的数目，W 是词汇表 words 中的词。

    ```
   示例 1：
   输入：queries = ["cbd"], words = ["zaaaz"]
   输出：[1]
   解释：查询 f("cbd") = 1，而 f("zaaaz") = 3 所以 f("cbd") < f("zaaaz")。
   
   示例 2：
   输入：queries = ["bbb","cc"], words = ["a","aa","aaa","aaaa"]
   输出：[1,2]
   解释：第一个查询 f("bbb") < f("aaaa")，第二个查询 f("aaa") 和 f("aaaa") 都 > f("cc")。
    ```


   提示：

   - 1 <= queries.length <= 2000
   - 1 <= words.length <= 2000
   - 1 <= queries[i].length, words[i].length <= 10
   - `queries[i][j], words[i][j] `都是小写英文字母

2. 简单实现

   ```c++
   class Solution {
   public:
       int count(string& s){//统计s中（按字典序比较）最小字母的出现频次。
           sort(s.begin(), s.end());
           char c = s[0];
           int ans = 0;
           for(int i = 0; i < s.size(); i++){
               if(s[i] == c)
                   ans++;
               else
                   break;
           }
           return ans;
       }
       vector<int> numSmallerByFrequency(vector<string>& queries, vector<string>& words) {
           int n = words.size();
           vector<int> cnt(n);
           for(int i = 0; i < n; i++)
               cnt[i] = count(words[i]);
           sort(cnt.begin(), cnt.end());//按顺序存储words内个单词的count值
           vector<int> ans(queries.size());
           for(int i = 0; i < queries.size(); i++){
               int num = count(queries[i]);
               ans[i] = cnt.end() - upper_bound(cnt.begin(), cnt.end(), num);
           }
           return ans;
       }
   };
   ```

### 32. 最长有效括号（困难）

---

1. 题目描述

   给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。

   ```
   示例 1:
   输入: "(()"
   输出: 2
   解释: 最长有效括号子串为 "()"
   
   示例 2:
   输入: ")()())"
   输出: 4
   解释: 最长有效括号子串为 "()()"
   ```

2. 简单实现

   动态规划，dp[i]表示以s[i]开头的最长有效括号串长度，则`dp[i] = j-i+1 + dp[j]`，其中s[i...j]为有效括号串，但是如果遍历找j会超时，需要使用trick：

   - 观察到，以(开始的有效括号串形式可能有`()xxx和(xxx)`，其中xxx为有效括号串
   - 因此，只需要找到与开头匹配的')'即可，中间过程可以根据已经处理的xxx的开头进行跳转

   ```c++
   class Solution {
   public:
       int longestValidParentheses(string s) {
           int len = s.size();
           vector<int> dp(len+1, 0);//len+1简化边界处理
           int idx = len - 2;
           int ans = 0;
           while(idx >= 0){//从后向前遍历
               if(s[idx] == '('){//只有'('有可能是有效括号串的起始字符
                   int r = idx+1;
                   while(r < len){//依次探查后面的字符，其实后面状态都是
                       if(s[r] == '('){
                           if(dp[r] == 0) break;//以s[r]开头无有效括号串，则以s[idx]开头必然也没有
                           r += dp[r];//跳过以s[r]开头的有效括号串，继续查看
                       }
                       else{//匹配到了')'
                           dp[idx] = r-idx+1 + dp[r+1];
                           break;
                       }
                   }
               }
               ans = max(ans, dp[idx]);
               idx--;
           }
           return ans;
       }
   };
   ```

3. 最优解法——无额外空间的双指针

   ![1589776219481](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1589776219481.png)

   ```java
   public class Solution {
       public int longestValidParentheses(String s) {
           int left = 0, right = 0, maxlength = 0;
           for (int i = 0; i < s.length(); i++) {
               if (s.charAt(i) == '(') 
                   left++;
               else 
                   right++;
               if (left == right)
                   maxlength = Math.max(maxlength, 2 * right);
               else if (right >= left)
                   left = right = 0;
           }
           left = right = 0;
           for (int i = s.length() - 1; i >= 0; i--) {
               if (s.charAt(i) == '(') 
                   left++;
               else
                   right++;
               if (left == right) 
                   maxlength = Math.max(maxlength, 2 * left);
               else if (left >= right)
                   left = right = 0;
           }
           return maxlength;
       }
   }
   ```

### 248. 中心对称数III（困难）

---

要会员

### 337. 打家劫舍III（中等）

---

1. 题目描述

   在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

   计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

       示例 1:
       输入: [3,2,3,null,3,null,1]
         3
        / \
       2   3
        \   \ 
         3   1
       输出: 7 
       解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
       示例 2:
       输入: [3,4,5,1,3,null,1]
          3
         / \
        4   5
         / \   \ 
        1   3   1
       输出: 9
       解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.

2. 正确解法

   一开始用递归写的代码如下所示，超时了，

   ```c++
   class Solution {
   public:
       int helper(TreeNode* root, bool f){//f表示父亲节点是否被偷
           if(!root) return 0;
           int re = helper(root->left, false) + helper(root->right, false);//不偷当前节点
           if(!f)
               re = max(re, root->val+helper(root->left, true)+helper(root->right, true));//偷
           return re;
       }
       int rob(TreeNode* root) {
           return helper(root, false);
       }
   };
   ```

   看别人答案如下逻辑就过了

   ```c++
   class Solution {
   public:
       vector<int> helper(TreeNode* root){//返回不偷/偷root节点的情况下能取得的最多钱
           if(!root) return {0,0};
           vector<int> re = {0,0};
           vector<int> l = helper(root->left);
           vector<int> r = helper(root->right);
           re[0] = max(l[0], l[1]) + max(r[0], r[1]);//不偷root
           re[1] = root->val + l[0] + r[0];//偷root
           return re;
       }
       int rob(TreeNode* root) {
           vector<int> ans = helper(root);
           return max(ans[0], ans[1]);
       }
   };
   ```

### 695. 岛屿的最大面积（中等）

---

1. 题目描述

   给定一个包含了一些 0 和 1 的非空二维数组 grid 。

   一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

   找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

   ```
   示例 1:
   [[0,0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,1,1,0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,1,0,1,0,0],
    [0,1,0,0,1,1,0,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0]]
   对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1 。
   
   示例 2:
   [[0,0,0,0,0,0,0,0]]
   对于上面这个给定的矩阵, 返回 0。
   ```

   注意: 给定的矩阵grid 的长度和宽度都不超过 50。

2. 简单实现——BFS

   ```c++
   class Solution {
   public:
       vector<vector<int>> dirs = {{0,1}, {0,-1}, {-1,0}, {1,0}};
       int m,n;
       int bfs(vector<vector<int>>& grid, int x, int y){//BFS计算grid[x][y]所属的岛屿面积
           queue<pair<int,int>> q;
           q.push({x, y});
           grid[x][y] = 0;
           int ans = 0;
           while(!q.empty()){
               int i = q.front().first;
               int j = q.front().second;
               q.pop();
               ans++;
               for(int k = 0; k < 4; k++){
                   int xx = i + dirs[k][0];
                   int yy = j + dirs[k][1];
                   if(xx >= 0 && xx < m && yy >= 0 && yy < n && grid[xx][yy] == 1){
                       grid[xx][yy] = 0;
                       q.push({xx, yy});
                   }
               }
           }
           return ans;
       }
       int maxAreaOfIsland(vector<vector<int>>& grid) {
           m = grid.size();
           n = grid[0].size();
           int ans = 0;
           for(int i = 0; i < m; i++)
               for(int j = 0; j < n; j++){
                   if(grid[i][j] == 1){
                       ans = max(ans, bfs(grid, i, j));
                   }
               }
           return ans;
       }
   };
   ```

### 134. 加油站（中等）

---

1. 题目描述

   在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

   你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

   如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

   说明: 

   - 如果题目有解，该答案即为唯一答案。
   - 输入数组均为非空数组，且长度相同。
   - 输入数组中的元素均为非负数。

   ```
   示例 1:
   输入: 
   gas  = [1,2,3,4,5]
   cost = [3,4,5,1,2]
   输出: 3
   解释:
   从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
   开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
   开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
   开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
   开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
   开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
   因此，3 可为起始索引。
   
   示例 2:
   输入: 
   gas  = [2,3,4]
   cost = [3,4,3]
   输出: -1
   解释:
   你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
   我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
   开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
   开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
   你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
   因此，无论怎样，你都不可能绕环路行驶一周。
   ```

2. 简单实现

   - 定义`dif[i] = gas[i] - cost[i]`，表示从当前站走到下一站后，要消耗/增加的油量
   - 如果从j处出发可以环形行驶一周，则一定有`sum(dif[j...k]) >= 0, j <= k <n`且`sum(dif[j...n]+sum[0...j)`成立，即从j开始逐站累加dif，和（油箱剩余油量）总为非负数
   - 可以想到，dif数组是由若干和连续的负数和若干个连续的正数交错构成的，由贪心的角度思考，满足条件的j值一定是某一段连续的正数中的第一个正数
   - 基于上述发现，用双指针遍历即可，具体见代码注释

   ```c++
   class Solution {
   public:
       int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
           int n = gas.size();
           vector<int> dif(n);
           for(int i = 0; i < n; i++)
               dif[i] = gas[i] - cost[i];
           int l = 0;///记录当前待判断起点
           int pre = 0;//l点前的dif和 
           while(l < n && dif[l] < 0){
               pre += dif[l];
               l++;
           }
           int r = l;//以l为起点，当前行驶到的位置
           while(r < n){
               int cur = 0;//记录当前油箱剩余油量
               while(r < n && cur >= 0){
                   cur += dif[r];
                   r++;
               }
               if(r == n && cur + pre >= 0)//满足条件
                   return l;
               else{//寻找下一个可能的起点l，即下一个连续正数段的起点
                   while(l < n && dif[l] >= 0)
                       pre += dif[l++];
                   while(l < n && dif[l] < 0)
                       pre += dif[l++];
                   r = l;
               }
           }
           return -1;
       }
   };
   ```

### 12. 整数转罗马数字（中等）

---

1. 题目描述

   罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

   ```
   字符          数值
   I             1
   V             5
   X             10
   L             50
   C             100
   D             500
   M             1000
   ```

   例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

   通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

   - I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
   - X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
   - C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。

   给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

   ```
   示例 1:
   输入: 3
   输出: "III"
   
   示例 2:
   输入: 4
   输出: "IV"
   
   示例 3:
   输入: 9
   输出: "IX"
   
   示例 4:
   输入: 58
   输出: "LVIII"
   解释: L = 50, V = 5, III = 3.
   
   示例 5:
   输入: 1994
   输出: "MCMXCIV"
   解释: M = 1000, CM = 900, XC = 90, IV = 4.
   ```

2. 简单实现

   ```c++
   class Solution {
   public:
       string intToRoman(int num) {
           string ans = "";
           int c = num / 1000;
           if(c > 0)
               ans += string(c, 'M');
           c = num % 1000 / 100;
           if(c > 0){
               if(c < 4)
                   ans += string(c, 'C');
               else if(c == 4)
                   ans += "CD";
               else if(c < 9)
                   ans += 'D' + string(c-5, 'C');
               else 
                   ans += "CM";
           }
           c = num % 100 / 10;
           if(c > 0){
               if(c < 4)
                   ans += string(c, 'X');
               else if(c == 4)
                   ans += "XL";
               else if(c < 9)
                   ans += 'L' + string(c-5, 'X');
               else 
                   ans += "XC";
           }
           
           c = num % 10;
           if(c > 0){
               if(c < 4)
                   ans += string(c, 'I');
               else if(c == 4)
                   ans += "IV";
               else if(c < 9)
                   ans += 'V' + string(c-5, 'I');
               else 
                   ans += "IX";
           }
           return ans;
       }
   };
   ```

### 45. 跳跃游戏II（困难）

---

1. 题目描述

   给定一个非负整数数组，你最初位于数组的第一个位置。

   数组中的每个元素代表你在该位置可以跳跃的最大长度。

   你的目标是使用最少的跳跃次数到达数组的最后一个位置。

   ```
   示例:
   输入: [2,3,1,1,4]
   输出: 2
   解释: 跳到最后一个位置的最小跳跃数是 2。
        从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
   ```

   说明: 假设你总是可以到达数组的最后一个位置。

2. 正确解法

   ![1589864544103](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1589864544103.png)

   ![1589864569941](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1589864569941.png)



   ```c++
   class Solution {
   public:
       int jump(vector<int>& nums) {
           int maxPos = 0, n = nums.size(), end = 0, step = 0;
           for (int i = 0; i < n - 1; ++i) {
               if (maxPos >= i) {
                   maxPos = max(maxPos, i + nums[i]);
                   if (i == end) {
                       end = maxPos;
                       ++step;
                   }
               }
           }
           return step;
       }
   };
   ```

### 350. 两个数组的交集II（简单）

---

1. 题目描述

   给定两个数组，编写一个函数来计算它们的交集。

   ```
   示例 1:
   输入: nums1 = [1,2,2,1], nums2 = [2,2]
   输出: [2,2]
   
   示例 2:
   输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
   输出: [4,9]
   ```

   说明：

   - 输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
   - 我们可以不考虑输出结果的顺序。

   进阶:

   - 如果给定的数组已经排好序呢？你将如何优化你的算法？
   - 如果 nums1 的大小比 nums2 小很多，哪种方法更优？
   - 如果 nums2 的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？

2. 简单实现

   ```c++
   class Solution {
   public:
   	vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
   		unordered_map<int, int> m;
           vector<int> ans;
           if(nums1.size() == 0 || nums2.size() == 0) return ans;
           //计数
           for(int i = 0; i < nums1.size(); i++){
               if(m.count(nums1[i]) <= 0)
                   m[nums1[i]] = 1;
               else{
                   m[nums1[i]] += 1;
               }
           }
           //查重
           for(int i = 0; i < nums2.size(); i++){
               if(m.count(nums2[i]) > 0 && m[nums2[i]] != 0){
                   ans.push_back(nums2[i]);
                   m[nums2[i]] -= 1;
               }
           }
           return ans;
   	}
   };
   ```

### 353. 贪吃蛇（中等）

---

要会员

### 360. 有序转化数组（中等）

---

要会员

### 363. 矩形区域不超过K的最大数值和（困难)

---

1. 题目描述

   给定一个非空二维矩阵 matrix 和一个整数 k，找到这个矩阵内部不大于 k 的最大矩形和。

   ```
   示例:
   输入: matrix = [[1,0,1],[0,-2,3]], k = 2
   输出: 2 
   解释: 矩形区域 [[0, 1], [-2, 3]] 的数值和是 2，且 2 是不超过 k 的最大数字（k = 2）。
   ```

   说明：

   - 矩阵内的矩形区域面积必须大于 0。
   - 如果行数远大于列数，你将如何解答呢？

2. 正确解法

   <https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/solution/javacong-bao-li-kai-shi-you-hua-pei-tu-pei-zhu-shi/>

   <https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/solution/bao-li-onyou-hua-er-fen-er-fen-jian-zhi-by-lzh_yve/>

   ```c++
   class Solution {
   public:
       int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
           int row=matrix.size();
           if (row==0) return 0;
           int column=matrix[0].size();
           int ans=INT_MIN;
           for (int left=0;left<column;++left)         {
               vector<int> row_sum(row,0);
               for (int right=left;right<column;++right) {
                   for (int i=0;i<row;++i)
                       row_sum[i] += matrix[i][right];
                   set<int> helper_set;
                   helper_set.insert(0);
                   int prefix_row_sum=0;
                   for (int i=0;i<row;++i) {
                       prefix_row_sum+=row_sum[i];
                       auto p=helper_set.lower_bound(prefix_row_sum-k);
                       helper_set.insert(prefix_row_sum);
                       if (p==helper_set.end())
                           continue;
                       else {
                           int temp=prefix_row_sum-(*p);
                           if (temp>ans)
                               ans=temp;
                       }
                   }
                   if (ans==k)
                       return k;
               }
           }        
           return ans;
       }
   };
   ```

### 400. 第N个数字（中等）

---

1. 题目描述

   在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 n 个数字。

   注意: n 是正数且在32位整数范围内 ( n < 231)。

   ```
   示例 1:
   输入: 3
   输出: 3
   
   示例 2:
   输入: 11
   输出: 0
   说明: 第11个数字在序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是0，它是10的一部分。
   ```

2. 简单实现

   找规律计算题：

   - 长度为1：1~9，共9个数，1*9个数字
   - 长度为2：10~99，共90个数，2*90个数字
   - 长度为3：100~999，共900个数，3*900个数字
   - 以此类推

   ```c++
   class Solution {
   public:
       int findNthDigit(int n) {
           int ans = 0;
           int pre = 0;
           for(int i = 1; i < 10; i++){//依次看第n个数字是否属于i位数
               if(n <= pre + i*9*pow(10, i-1)){//属于
                   int base = pow(10, i-1);
                   int aim = base + (n-pre-1) / i;//第n个数组所在的整数
                   int cur = pre + i*(aim-base) + 1;//aim的第一个数字是整个序列的第cur个数字
                   string s = to_string(aim);
                   int idx = 0;
                   while(cur < n){//找到第n个数字
                       idx++;
                       cur++;
                   }
                   return s[idx]-'0';
               }
               pre += i*9*pow(10, i-1);
           }
           return ans;
       }
   };
   ```

### 562. 矩阵中最长的连续1线段（中等）

---

要会员

### 946. 验证栈序列（中等）

---

1. 题目描述

   给定 pushed 和 popped 两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。

   ```
   示例 1：
   输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
   输出：true
   解释：我们可以按以下顺序执行：
   push(1), push(2), push(3), push(4), pop() -> 4,
   push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
   
   示例 2：
   输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
   输出：false
   解释：1 不能在 2 之前弹出。
   ```


   提示：

   - 0 <= pushed.length == popped.length <= 1000
   - 0 <= pushed[i], popped[i] < 1000
   - pushed 是 popped 的排列。

2. 简单实现——模拟过程

   ```c++
   class Solution {
   public:
       bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
           stack<int> s;
           int idx1 = 0, idx2 = 0;
           int size = pushed.size();
           while(idx1 < size){
               while(idx1 < size && (s.empty() || s.top() != popped[idx2])){
                   s.push(pushed[idx1]);
                   idx1++;
               }
               while(idx2 < size && !s.empty() && s.top() == popped[idx2]){
                   s.pop();
                   idx2++;
               }
           }
           return idx2 == size;
       }
   };
   ```

### 161. 相隔为1的编辑距离（中等）

---

要会员

### 28. 实现strStr()（简单）

---

1. 题目描述

   实现 strStr() 函数。

   给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

   ```
   示例 1:
   输入: haystack = "hello", needle = "ll"
   输出: 2
   
   示例 2:
   输入: haystack = "aaaaa", needle = "bba"
   输出: -1
   ```

   说明：当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

   对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

2. 简单实现

   KMP算法

   ```c++
   class Solution {
   public:
       void get_next(string s, vector<int> &next){
           next[0] = 0;
           int j = 0;
           int i = 1;
           while(i < s.size()){
               if(s[i] == s[j]){
                   next[i] = j + 1;
                   i++;
                   j++;
               }
               else{
                   if(j == 0){
                       next[i] = 0;
                       i++;
                   }
                   else
                       j = next[j-1];
               }
           }
       }
       int strStr(string haystack, string needle) {
           if(needle == "")
               return 0;
           vector<int> next = vector<int>(needle.size(), 0);
           get_next(needle, next);
           int i = 0;
           int j = 0;
           while(i < haystack.size() && j < needle.size()){
               if(haystack[i] == needle[j]){
                   i++;
                   j++;
               }
               else{
                   if(j != 0)
                       j = next[j-1];
                   else
                       i++;
               }
           }
           if(j == needle.size())
               return i - needle.size();
           else
               return -1;
       }
   };
   ```

### 71. 简化路径（中等）

---

1. 题目描述

   以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。

   在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。更多信息请参阅：Linux / Unix中的绝对路径 vs 相对路径

   请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。

    ```
   示例 1：
   输入："/home/"
   输出："/home"
   解释：注意，最后一个目录名后面没有斜杠。
   
   示例 2：
   输入："/../"
   输出："/"
   解释：从根目录向上一级是不可行的，因为根是你可以到达的最高级。
   
   示例 3：
   输入："/home//foo/"
   输出："/home/foo"
   解释：在规范路径中，多个连续斜杠需要用一个斜杠替换。
   
   示例 4：
   输入："/a/./b/../../c/"
   输出："/c"
   
   示例 5：
   输入："/a/../../b/../c//.//"
   输出："/c"
   
   示例 6：
   输入："/a//b////c/d//././/.."
   输出："/a/b/c"
    ```

2. 简单实现

   ```c++
   class Solution {
   public:
       string simplifyPath(string path) {
           deque<string> s;
           string cur = "";
           int len = path.size();
           int idx = 0;
           while(idx < len){
               if(path[idx] == '/'){
                   if(cur == "." || cur == "")
                       cur = "";
                   else if(cur == ".."){
                       if(!s.empty())
                           s.pop_back();
                   }
                   else
                       s.push_back(cur);
                   cur = "";
               }
               else
                   cur += path[idx];
               idx++;
           }
           if(cur == "." || cur == "")
               cur = "";
           else if(cur == ".."){
               if(!s.empty())
                   s.pop_back();
           }
           else
               s.push_back(cur);
           cur = "";
           if(s.empty())
               return "/";
           string ans = "";
           while(!s.empty()){
               ans += '/' + s[0];
               s.pop_front();
           }
           return ans;
       }
   };
   ```

### 387. 字符串中第一个唯一的字符（简单）

---

1. 题目描述

   给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

   ```
   案例:
   s = "leetcode"
   返回 0.
   
   s = "loveleetcode",
   返回 2.
   ```


   注意事项：您可以假定该字符串只包含小写字母。

2. 简单实现

   ```c++
   class Solution {
   public:
       int firstUniqChar(string s) {
           vector<int> cnt(26, 0);
           for(int i = 0; i < s.size(); i++)
               cnt[s[i]-'a']++;
           for(int i = 0; i < s.size(); i++)
               if(cnt[s[i]-'a'] == 1)
                   return i;
           return -1;
       }
   };
   ```

### 498. 对角线遍历（中等）

---

1. 题目描述

   给定一个含有 M x N 个元素的矩阵（M 行，N 列），请以对角线遍历的顺序返回这个矩阵中的所有元素，对角线遍历如下图所示。

   示例:

   输入:
   [
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]
   ]

   输出:  [1,2,4,7,5,3,6,8,9]

   解释:

    ![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/diagonal_traverse.png)

   说明: 给定矩阵中的元素总数不会超过 100000 。

2. 简单实现

   ```c++
   class Solution {
   public:
       vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
           vector<int> ans;
           int m = matrix.size();
           if(m == 0)
               return ans;
           int n = matrix[0].size();
           
           int dire[2] = {-1, 1};
           int x = 0, y = 0;
           while(1){
               ans.push_back(matrix[x][y]);
               if(x == m-1 && y == n-1)
                   break;
               if(dire[0] < 0){//左下
                   if(y == n-1){
                       x++;
                       dire[0] = -dire[0];
                       dire[1] = -dire[1];
                   }
                   else if(x == 0){
                       y++;
                       dire[0] = -dire[0];
                       dire[1] = -dire[1];
                   }
                   else{
                       x += dire[0];
                       y += dire[1];
                   }
               }
               else{//右上
                   if(x == m-1){
                       y++;
                       dire[0] = -dire[0];
                       dire[1] = -dire[1];
                   }
                   else if(y == 0){
                       x++;
                       dire[0] = -dire[0];
                       dire[1] = -dire[1];
                   }
                   else{
                       x += dire[0];
                       y += dire[1];
                   }
               }
           }
           return ans;
       }
   };
   ```

### 774. 最小化去加油站的最大距离（困难）

---

要会员

### 1087. 字母切换（中等）

---

要会员

### 25. K个一组翻转链表（困难）

---

1. 题目描述

   给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

   k 是一个正整数，它的值小于或等于链表的长度。

   如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

    ```
   示例：
   给你这个链表：1->2->3->4->5
   当 k = 2 时，应当返回: 2->1->4->3->5
   当 k = 3 时，应当返回: 3->2->1->4->5
    ```

   说明：

   - 你的算法只能使用常数的额外空间。
   - 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

2. 简单实现

   ```c++
   class Solution {
   public:
       ListNode* reverseKGroup(ListNode* head, int k) {
           ListNode* ans = new ListNode(-1);
           ans->next = head;
           ListNode* begin = ans;//指向当前段开头的前一个节点
           ListNode* end = ans;//用于看当前段够不够k长
           while(end){
               for(int i = 0; end && i < k; i++){
                   end = end->next;
               }
               if(!end) break;//不够k,不翻转
               //翻转begin->next到end之间的k个节点
               ListNode* pre = begin;
               ListNode* cur = begin->next;
               for(int i = 0; i < k; i++){
                   ListNode* tmp = cur->next;
                   cur->next = pre;
                   pre = cur;
                   cur = tmp;
               }
               //处理当前段的首尾
               ListNode* tmp = begin->next;
               begin->next->next = cur;//尾端与下一段相接
               begin->next = pre;//头部与上一段相接
               begin = tmp;//更新
               end = tmp;//更新
           }
           return ans->next;
       }
   };
   ```

### 100. 相同的树（简单）

---

1. 题目描述

   给定两个二叉树，编写一个函数来检验它们是否相同。

   如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

       示例 1:
       输入:       1         1
                 / \       / \
                2   3     2   3
           
       [1,2,3],   [1,2,3]
       输出: true
    
       示例 2:
       输入:      1          1
                 /           \
                2             2
                
       [1,2],     [1,null,2]
       输出: false
    
       示例 3:
       输入:       1         1
                 / \       / \
                2   1     1   2
       
       [1,2,1],   [1,1,2]
       输出: false

2. 简单实现

   ```c++
   class Solution {
   public:
       bool isSameTree(TreeNode* p, TreeNode* q) {
           if(!p && !q) return true;
           if(!p || !q) return false;
           if(p->val == q->val)    
               return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
           else return false;
       }
   };
   ```

### 120. 三角形最小路径和（中等）

---

1. 题目描述

   给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

   相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。

   ```
   例如，给定三角形：
   [
        [2],
       [3,4],
      [6,5,7],
     [4,1,8,3]
   ]
   自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
   ```

   说明：如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。

2. 简单实现

   动态规划，修改了原数组，没用额外空间，修改后`triangle[i][j]`表示自顶向下走到i行j列的最小路径和

   ```c++
   class Solution {
   public:
       int minimumTotal(vector<vector<int>>& triangle) {
           int m = triangle.size();
           for(int i = 1; i < m; i++){
               int n = triangle[i].size();
               triangle[i][0] += triangle[i-1][0];
               triangle[i][n-1] += triangle[i-1][n-2];
               for(int j = 1; j < n-1; j++){
                   triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j]);
               }
           }
           int ans = INT_MAX;
           for(int i = 0; i < triangle[m-1].size(); i++)
               ans = min(ans, triangle[m-1][i]);
           return ans;
       }
   };
   ```

   其实不用额外空间也行，因为`triangle[i][j]`只与上一行有关，而最后一行元素个数和行数n相等，所以只维护一行的动态规划，就是O(n)的额外空间

### 137. 只出现一次的数字II（中等）

---

1. 题目描述

   给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

   说明：你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

   ```
   示例 1:
   输入: [2,2,3,2]
   输出: 3
   
   示例 2:
   输入: [0,1,0,1,0,1,99]
   输出: 9
   ```

2. 正确解法

   看了大佬题解，以后遇到类似的题目都可以用数字电路的知识做，神奇

   <https://leetcode-cn.com/problems/single-number-ii/solution/luo-ji-dian-lu-jiao-du-xiang-xi-fen-xi-gai-ti-si-l/>

   ![1590378649316](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1590378649316.png)

   ![1590378680103](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1590378680103.png)

   ![1590378715204](C:\Users\surface\AppData\Roaming\Typora\typora-user-images\1590378715204.png)

   推广：其实有了这个解法，所有关于数值状态转移的题都可以用类似的解法，列出真值表，求出逻辑表达式，即可很容易写出程序。

### 217. 存在重复元素（简单）

---

1. 题目描述

   给定一个整数数组，判断是否存在重复元素。

   如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。

    ```
   示例 1:
   输入: [1,2,3,1]
   输出: true
   
   示例 2:
   输入: [1,2,3,4]
   输出: false
   
   示例 3:
   输入: [1,1,1,3,3,4,3,2,4,2]
   输出: true
    ```

2. 简单实现

   ```c++
   class Solution {
   public:
       bool containsDuplicate(vector<int>& nums) {
           unordered_set<int> s;
           for(int i = 0; i < nums.size(); i++){
               if(s.count(nums[i]) > 0)
                   return true;
               s.insert(nums[i]);
           }
           return false;
       }
   };
   ```

### 273. 整数转换英文表示（困难）

---

1. 题目描述

   将非负整数转换为其对应的英文表示。可以保证给定输入小于 231 - 1 。

   ```
   示例 1:
   输入: 123
   输出: "One Hundred Twenty Three"
   
   示例 2:
   输入: 12345
   输出: "Twelve Thousand Three Hundred Forty Five"
   
   示例 3:
   输入: 1234567
   输出: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
   
   示例 4:
   输入: 1234567891
   输出: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
   ```

2. 简单实现

   ```c++
   class Solution {
   public:
       vector<string> dic1 = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",  "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
       vector<string> dic2 = {"","","Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
       string get3len(int num){
           string ans = "";
           if(num < 20)
               return dic1[num];
           else if(num < 100){
               ans += dic2[num/10];
               num %= 10;
               if(num > 0)
                   ans += " " + get3len(num);
           }
           else{
               ans += dic1[num/100] + " Hundred";
               num %= 100;
               if(num > 0)
                   ans += " " + get3len(num);
           }
           return ans;
       }
       string numberToWords(int num) {
           string ans = "";
           if(num < 1000)
               return get3len(num);
           else if(num < 1000000){
               ans += get3len(num / 1000) + " Thousand";
               num %= 1000;
               if(num > 0)
                   ans += " " + numberToWords(num);
           }
           else if(num < 1000000000){
               ans += get3len(num / 1000000) + " Million";
               num %= 1000000;
               if(num > 0)
                   ans += " " + numberToWords(num);
           }
           else{
               ans += get3len(num / 1000000000) + " Billion";
               num %= 1000000000;
               if(num > 0)
                   ans += " " + numberToWords(num);
           }
           return ans;
       }
   };
   ```

### 325. 和等于K的最长子数组长度（中等）

---

要会员