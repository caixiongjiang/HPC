## MIT 6.172

### Introduction

> 什么是摩尔定律？

摩尔定律：集成电路上可容纳的晶体管数量每隔约18至24个月就会翻倍，而其成本则保持不变。

随着晶体管数量的增加，计算机的处理能力也随之提升。这使得计算机的速度更快、尺寸更小、功耗更低，为信息技术的发展奠定了基础。



### Matrix Multiplication

$C=A\times B$，其中$A$，$B$，$C$都是$n\times n$的矩阵，理论上它需要大约$2n^3$。



> Python的矩阵乘法实现

Code:

```python
import sys, random
from time import *

n = 4096

A = [[random.random() for row in range(n)]
    							for col in range(n)]
B = [[random.random() for row in range(n)]
    							for col in range(n)]
C = [[random.random() for row in range(n)]
    							for col in range(n)]
start = time()
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]
end = time()

print ('%0.6f').format(end - start)
```

*注意这段代码的运行在峰值性能为`836GFLOPs`的计算机上大约需要6个小时。*

浮点计算次数 = $2n^3 = 2(2^{12})^3 = 2^{37}$

Runtime = $21042s$

Python得到的计算峰值性能为$2^{37}/21042\approx 6.25 \text{MFLOPs}$，大约为峰值性能的0.00075%



> Java的矩阵乘法实现

Code:

```java
import java.util.Random;

public class mm_java { 
  	static int n = 4096;
		static double[][] A = new double[n][n];
		static double[][] B = new double[n][n];
  	static double[][] C = new double[n][n];

		public static void main(String[] args) {
      	Random r = new Random();

				for (int i=0; i<n; i++) {
						for (int j=0; j<n; j++) {
              A[i][j] = r.nextDouble();
              B[i][j] = r.nextDouble();
              C[i][j] = 0;
            }
				}
      	long start = System.nanoTime();
      	
      	for (int i=0; i<n; i++) {
          	for (int j=0; j<n; j++) {
              	for (int k=0; k<n; k++) {
                  	C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
      	long stop = System.nanoTime();
      	
      	double tdiff = (stop - start) * 1e-9;
      	System.out.println(tdiff);
    }
}
```

*注意这段代码的运行在峰值性能为`836GFLOPs`的计算机上大约需要46分钟。*

这相比Python的实现将近快了9倍



> c的矩阵乘法实现

Code：

```c
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define n 4096
double A[n][n];
double B[n][n];
double C[n][n];

float tdiff(struct timeval *start,
						struct timeval *end) {
		return (end->tv_sec - start->tv_sec) + 
      1e-6*(end->tv_usec - start->tv_usec);
}
int main(int argc, const char *argv[]) {
		for (int i=0; i<n; ++i) {
				for (int j=0; j<n; ++j) {
          	for (int k=0; k<n; ++k) {
              	A[i][j] = (double)rand() / (double)RAND_MAX;
              	B[i][j] = (double)rand() / (double)RAND_MAX;
              	C[i][j] = 0;
            }
        }
    }
		
  	struct timeval start, end;
  	gettimeofday(&start, NULL);
  
  	for (int i=0; i<n; ++i) {
      	for (int j=0; j<n; ++j) {
          	for (int k=0; k<n; ++k) {
              C[i][j] = A[i][k] + B[k][j];
            }
        }
    }
  
  	gettimeofday(&end, NULL);
  	print("%0.6f\n", tidff(&start, &end));
    return 0;
}
```

*注意这段代码的运行在峰值性能为`836GFLOPs`的计算机上大约需要19分钟。*

它的速度是Java的2倍，Python的18倍。



它们的性能对比图如下所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img8.jpg)

造成这个结果的原因是：

* Python是解释型语言
* C语言会直接被编译成机器语言
* Java则先被编译为字节码，然后JIT编译成机器语言



> 改变循环的顺序不会改变执行的结果，但程序运行的速度是否有变化？

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img9.jpg)

矩阵在内存中是按照顺序排列的，执行顺序是行优先还是列有限，对执行快慢由很大的影响。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img10.jpg)

*如上图所示，访问矩阵B的列元素时，每个元素分布在不同的位置（cache），访问很耗时（空间局部性很差）。*

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img11.jpg)

*可以看到，修改顺序后，访问内存的效率立马变得高效起来了。执行的时间变为了177.68秒*

* 使用`Cachegrind缓存模拟器`测量不同访问模式的效果：

```shell
$ valgrind --tool=cachegrind ./mm
```



> 除了改变顺序，我们还可以如何加速程序？

`Clang`提供了一组优化开关。可以指定编译器的切换，以要求其进行优化。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img12.jpg)



> 上述所有操作都是在单核下进行的优化，接下来我们还可以使用多个核心去完成这件事情。

`cilk_for`循环允许循环的所有迭代并行执行：

```c
cilk_for (int i = 0; i < n; ++i)
  	for (int k = 0; k < n; ++k)
      	cilk_for (int j = 0; j < n; ++j) 
      			C[i][j] += A[i][k] * B[k][j];
```



下图是三种并行方式的运行时间：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img13.jpg)



> 为什么使用了并行计算，有时反而会增加计算时间呢？

这与系统的调度开销有关，这里的经验法则是**并行化外循环**。使用了正确的并行，我们又将性能提升了18倍。



> 经过上述优化之后，只达到了机器峰值性能的5%，什么使得程序变得这么慢？

回到硬件的层面，又一个东西被我们忽略了，那就是缓存命中率（重用数据）。



* 计算C中的某一行（4096个数），需要便利A中的某一行和B中的所有列：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img14.jpg)



* 计算C中的某个块（64*64），而相比某一行，我们访问内存的次数大大减少了:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img15.jpg)



> 从上述两个图片，我们可以使用什么策略继续加速矩阵乘法呢？

可以将矩阵分成若干个子矩阵分别计算，程序如下：

```c
cilk_for (int ih = 0; ih < n; ih += s)
  	cilk_for (int jh = 0; jh < n; jh += s)
  			for (int kh = 0; kh < n; kh += s)
          	for (int il = 0; il < s; ++il)
              	for (int kl = 0; kl < s; ++kl)
                  	for (int jl = 0; jl < s; ++jl)
                      	C[ih+il][jh+jl] += A[ih+il][kh+kl] * B[kh+kl][jh+jl];
```



![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img16.jpg)

*根据不同的s取值，运行时间不同，如上图所示*



> 计算机中存在三级缓存，为了更好地使用缓存，我们可以使用递归喝分治的思想，让每一个矩阵往下分解矩阵执行。



```c
void mm_dac(double *restrict C, int n_C,
            double *restrict A, int n_A,
            double *restrict B, int n_B,
            int n)
{ // C += A * B
		assert((n & (-n)) == n);
		if (n <= 1) { //1应该改为THRESHOLD，用于改变最小矩阵的大小
      	*C += *A * *B;
    } else {
    		#define X(M,r,c) (M + (r*(n_ ## M) + c)*(n/2))
      			// cilk_spawn 子函数调用被生成，这意味着它可以与父调用者并行执行。
      			cilk_spawn mm_dac(X(C, 0, 0), n_C, X(A,0,0), n_A, X(B,0,0), n_B, n/2);
      			cilk_spawn mm_dac(X(C, 0, 1), n_C, X(A,0,0), n_A, X(B,0,1), n_B, n/2);
      			cilk_spawn mm_dac(X(C, 1, 0), n_C, X(A,1,0), n_A, X(B,0,0), n_B, n/2);
      								 mm_dac(X(C, 1, 1), n_C, X(A,1,0), n_A, X(B,0,1), n_B, n/2);
      			// cilk_sync 上述子函数返回结果之前，阻塞，防止下阶段发生。
      			cilk_sync;
      			cilk_spawn mm_dac(X(C, 0, 0), n_C, X(A,0,1), n_A, X(B,1,0), n_B, n/2);
      			cilk_spawn mm_dac(X(C, 0, 1), n_C, X(A,0,1), n_A, X(B,1,1), n_B, n/2);
      			cilk_spawn mm_dac(X(C, 1, 0), n_C, X(A,1,1), n_A, X(B,1,0), n_B, n/2);
      								 mm_dac(X(C, 1, 1), n_C, X(A,1,1), n_A, X(B,1,1), n_B, n/2);
      			cilk_sync;
    }
}
```

*这段代码的运行时间是93.93s，反而比之前更慢了。主要的原因是我们在不合适的时机仍然使用了子矩阵的方法（比如1$\times $1的的矩阵），程序的启动开销占了大部分时间*

所以要改变阈值，更改最小子矩阵的大小，分别计算性能，发现性能提升是很明显的(缓存的命中率提升很大)：

<img src="https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img17.jpg" alt="scale" style="zoom:50%;" />



**除了上述操作，我们还可以通过矢量操作来加速计算，SIMD；也可以通过改变计算顺序，虽然它输出的结果不是完全正确的，但可以提升速度。**



### Bentley Rules for Optimizing Work



#### 数据结构的优化

> 包装的想法是在一个机器单词中存储多个数据值。编码的相关想法是将数据值转换为需要更少位的表示。它们是互为相反的操作。

Example1:假设我们只存储在4096 B.C.和4096 C.E.之间的年份，大约有365.25 × 8192 ≈ 3 M日期，可以编码为⎡lg(3×10 6 )⎤ = 22位，很容易拟合单个（32位）单词。

Example2（Packing dates）:

```c
typedef struct {
  	int year: 13;
  	int month: 4;
 		int day: 5;
} date_t;
```



> 数据结构增强的想法是将信息添加到数据结构中，以使常见操作减少工作量。

Example1:单链表附加问题

如果一个单链表想要附加另一个链表，那么需要先找到链表在最后的位置，非常耗时。做法就是在单链表上额外存储一个尾指针，这样想要附加链表就非常容易：**只需要通过尾指针找到列表中的最后一个元素，最后一个元素的后继指针更改为指向第二个链表的头部，并将尾指针指向第二个链表的尾部。**

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img18.jpg)



> 预计算的想法是提前进行计算，以避免在“关键任务”时进行计算。

Example:
$$
\frac{n!}{k!(n-k)!}
$$
存储二项式系数的表叫`帕斯卡三角形`, 横坐标代表n，纵坐标代表k。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img19.jpg)

我们的C程序（生成帕斯卡三角形）可以写为：

```c
int choose(int n, int k) {
  	if (n < k) return 0;
  	if (n == 0) return 1;
  	if (k == 0) return 1;
  	return choose(n - 1, k - 1) + choose(n - 1, k);
}
```

如何对该函数进行预计算呢？其实就是提前计算$100\times 100$矩阵里的数。

```c
# define CHOOSE_SIZE 100
int choose[CHOOSE_SIZE][CHOOSE_SIZE];

void init_choose() {
  	for (int n = 0; n < CHOOSE_SIZE; ++n) {
      	choose[n][0] = 1;
      	choose[n][n] = 1;
    }
  	for (int n = 1; n < CHOOSE_SIZE; ++n) {
      	choose[0][n] = 0;
      	for (int k = 1; k < n; ++k) {
          	choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
          	choose[k][n] = 0;
        }
    }
}
```

该段程序只是在程序运行后先进行初始化，假设我们要多次运行该程序，还是会非常耗时。

> 如何进行多次计算只需要进行一次初始化呢？

答案是在编译时进行初始化，节省运行时时间。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img20.jpg)

> 如何预先编译$1000\times 1000$的表呢？

那么接下来就需要使用代码来写代码，也就是所谓的`元编程`。

```c
// 生成含有大量常量的表格，较为实用的技巧（注意元编程可以使用任意的语言，可以使用python更为简洁）
int main(int argc, const char *argv[]) {
  	init_choose();
  	printf("int choose[10][10] = {\n");
  	for (int a = 0; a < 10; ++a) {
      	printf(" {");
      	for (int b = 0; b < 10; ++b) {
          	printf("%3d, " choose[a][b]);
        }
      	printf("},\n");
    }
  	printf("};\n");
}
```



> 缓存的想法是存储最近访问的结果，以便程序无需再次计算它们。

Example:

看这一段程序:

```c
inline double hypotenuse(double A, double B) {
  	return sqrt(A*A + B*B);
}
```

调用计算`sqrt()`是很耗时的，那么我们可以创建缓存来减少计算。

```c
double cached_A = 0.0;
double cached_B = 0.0;
double cached_C = 0.0;

inline double hypotenuse(double A, double B) {
  	if (A == cached_A && B == cached_B) {
      	return cached_h;
    }
  	cached_A = A;
  	cached_B = B;
  	cached_h = sqrt(A*A + B*B);
  	return cached_h;
}
```

如果我遇到了前面重复的计算，便可以直接得到缓存的值，而不需要进行额外的计算了。



> 利用稀疏性的想法是避免在零上做存储和计算。（最快的方法是根本不计算）

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img21.jpg)

如上图所示，有许多零，其实都不需要进行计算。一种简单的优化方法是每次运行前检查一下是否为0，再决定是否需要进行计算，但其还是要遍历整个矩阵，判断是否需要计算。

> 我们可以用压缩稀疏行的结构来避免这种检查

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img22.jpg)

*获取每行元素的长度，可以直接通过偏移量的计算来确定。*

代码如下：

```c
typedef struct {
  	int n, nnz;
  	int *rows;  //length n
  	int *cols;	//length nnz
  	double *vals;//length nnz
} sparse_matrix_t;

void spmv(sparse_matrix_t *A, double *x, double *y) {
  	for (int i = 0; i < A->n; i++) {
      	y[i] = 0;
      	for (int k = A->rows[i]; k < A->rows[i + 1]; k++) {
          	int j = A->cols[k];
          	y[i] += A->vals[k] * x[j]; //执行矩阵乘法
        }
    }
}
```

*需要注意的是上面的解法只在含零较多的洗漱矩阵较为实用。*

> 对于稀疏图的存储，上述方法也可以使BFS等算法运行地更快。



#### 逻辑上的优化

> 恒定折叠和传播的想法是在编译期间评估常量表达式，并将结果替换为进一步的表达式。

Example:

```c
#include <math.h>

void orrey() {
  	const double radius = 6371000.0;
  	const double diameter = 2 * radius;
 		const double circumference = M_PI * diameter;
  	const double cross_area = M_PI * radius * radius;
  	const double surface_area = circumference * diameter;
  	const double volume = 4 * M_PI * radius * radius * radius / 3;
  	// ...
}
```

*在足够高的优化水平下，所有表达式都会在编译时进行评估。*



> 消除共同子表达式的想法是通过评估一次表达式并存储结果供以后使用来避免多次计算相同的表达式。

Example:

```scss
a = b + c;
b = a - d;
c = b + c;
d = a - d;
消除之后：
a = b + c;
b = a - d;
c = b + c;
d = b;
```



> 利用代数恒等式的想法是将昂贵的代数表达式替换为需要较少工作的代数等价物。

Example:

下面是检测两个球是否相撞的程序：

```c
#include <stdbool.h>
#include<math.h>

typedef struct {
  	double x;// x- coordinate
		double y;// y-coordinate
		double z;// z-coordinate
		double r;// radius of ball ballt;
}

double square(double x) { 
  	return x*x;
}

bool collides(ball_t *b1, ball_t *b2) {
		double d = sqrt(square(b1->x - b2->x)
						 + square(b1->y - b2->y) 
             + square(b1->z - b2->z));
		return d <= b1->r + b2->r; 	 
}
```

*事实证明，求平方根的计算是非常耗时的，我们只需要对判断的不等式两边求平方就能简化计算。*

程序变为：

```c
#include <stdbool.h>
#include<math.h>

typedef struct {
  	double x;// x- coordinate
		double y;// y-coordinate
		double z;// z-coordinate
		double r;// radius of ball ballt;
}

double square(double x) { 
  	return x*x;
}

bool collides(ball_t *b1, ball_t *b2) {
		double dsquared = square(b1->x - b2->x)
						 			  + square(b1->y - b2->y) 
                    + square(b1->z - b2->z);
		return dsquared <= square(b1->r + b2->r); 	 
}
```



> 在进行一系列测试时，短路的想法是，一旦您知道答案，就立即停止评估。

判断A的元素之和是否大于某个限制的值，A中的元素全部大于0:

```c
#include <stdbool.h>
//All elements of A are nonnegative
bool sum_exceeds(int *A, int n, int limit) {
  	int sum = 0;
  	for (int i = 0; i < n; i++) {
      	sum += A[i];
    }
  	return sum > limit;
}
```

*上述程序其实可以加一个判断，使得程序能够提前判断是否超过了限制的值。具体加入判断后是否进行了优化需要根据具体的例子来看，可以进行多次测试来说明是否适合此种优化。*

```c
#include <stdbool.h>
//All elements of A are nonnegative
bool sum_exceeds(int *A, int n, int limit) {
  	int sum = 0;
  	for (int i = 0; i < n; i++) {
      	sum += A[i];
      	if (sum > limit) {
          	return true;
        }
    }
  	return sum > limit;
}
```



> 考虑执行一系列逻辑测试的代码。订购测试的想法是在很少成功的测试之前执行那些更经常“成功”的测试——测试选择特定的替代方案。同样，便宜的测试应该先于昂贵的测试。

Example:

```c
#include <stdbool.h>

bool is_whitespace(char c) {
  	if (c == '\r' || c == '\t' || c == ' ' || c == '\n') {
      	return true;
    }
  	return false;
}
```

*上面的程序空格是更频繁的，逻辑或是短路运算符，如果左边符合条件则不会执行右边，所以左边需要放执行更频繁的逻辑判断。*

程序变为：

```c
#include <stdbool.h>
//只有为空格时能减少测试的工作量
bool is_whitespace(char c) {
  	if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
      	return true;
    }
  	return false;
}
```



> 创造一条快速路径

重新考虑上述判断球体碰撞的程序，我们可以通过球体的边界框提前知道没有相撞的情况，减少这部分的计算量。

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img23.jpg)



> 合并测试的想法是将一系列测试替换为一个测试或开关。

Example:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img24.jpg)



![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img25.jpg)

*当然你可以将这些表的内容先计算出来，然后每次测试都不需要重复计算了。*



#### 循环中的优化



> 提升（也称为循环不变代码运动）的目标是避免每次通过循环主体重新计算循环不变代码。

Example：

```c
#include <math.h>

void scale(double *X, double *Y, int N) {
  	for (int i = 0; i < N; i++) {
      	Y[i] = X[i] * exp(sqrt(M_PI/2));
    }
}
```

将其中的相同的运算表达式结果记录下来，这样只需要计算一次。

```c
#include <math.h>

void scale(double *X, double *Y, int N) {
  	double factor = exp(sqrt(M_PI/2));
  	for (int i = 0; i < N; i++) {
      	Y[i] = X[i] * factor;
    }
}
```



> 哨兵是放置在数据结构中的特殊虚拟值，以简化边界条件的逻辑，特别是循环退出测试的处理。

Example:

```c
#include <stdint.h>
#icnlude <stdbool.h>
//检查值是否溢出
bool overflow(int64_t *A, size_t n) {
		//All elements of A are nonnegative
  	int64_t sum = 0;
  	for (size_t i = 0; i < n; ++i) {
      	sum += A[i];
      	if (sum < A[i]) return true;//溢出时sum的值会迅速变为负值。
    }
  	return false;
}
```

程序可以优化为：

```c
#include <stdint.h>
#icnlude <stdbool.h>
//检查值是否溢出
bool overflow(int64_t *A, size_t n) {
		//All elements of A are nonnegative
  	A[n] = INT64_MAX;
  	A[n + 1] = 1; //或者设置称为任意一个正数
  	size_t i = 0;
  	int64_t sum = A[0];
  	while (sum >= A[i]) {
      	sum += A[++i];
    }
  	if (i < n) return true;
  	return false;
}
```



> 循环展开试图通过将循环的几个连续迭代合并为单个迭代来保存工作，从而减少循环的迭代总数，从而减少控制循环的指令必须执行的次数。

循环展开有两种方式：

* 全循环展开:

Example:

```c
int sum = 0;
for (int i = 0; i < 10; i++) {
  	sum += A[i];
}
```

在这种循环较小的时候，编译器可能会将其全部展开，而不用每次都去判断是否超过了循环的边界。

而比较常见的循环展开是部分循环展开：

```c
int sum = 0;
int j;
for (int j = 0; j < n - 3; j += 4) {
  	sum += A[j];
  	sum += A[j + 1];
  	sum += A[j + 2];
  	sum += A[j + 3];
}
//处理剩余的不能被4整除的数
for (int i = j; i < n; ++i) {
  	sum += A[i];
}
```

*这种优化主要思想是第一减少循环中判断（循环退出检查）的次数，第二为编译器提供了更多的优化空间，因为它增加了循环体的大小（更为主要）。但如果本身循环体就很大，再增加循环体的大小会导致污染指令集，反而会降低性能。*

* 循环融合：

循环融合（也称为干扰）的想法是将同一索引范围内的**多个循环组合成一个循环体**，从而节省循环控制的开销。

Example:

```c
for (int i = 0; i < n; ++i) {
  	C[i] = (A[i] < B[i]) ? A[i] : B[i];
}

for (int i = 0; i < n; ++i) {
  	D[i] = (A[i] <= B[i]) ? B[i] : A[i];
}
```

融合后：

```c
for (int i = 0; i < n; ++i) {
  	C[i] = (A[i] < B[i]) ? A[i] : B[i];
  	D[i] = (A[i] <= B[i]) ? B[i] : A[i];
}
```

*这既减少了循环条件检查的次数，更重要的是这增加了缓存击中率，使代码有更好的缓存局部性。*



> 消除浪费迭代的想法是修改循环边界，以避免在本质上空的循环体上执行循环迭代。

Example:

```c
for (int i = 0; i < n; ++i) {
  	for (int j = 0; j < n; ++j) {
      	if (i > j) {
          	int temp = A[i][j];
          	A[i][j] = A[j][i];
          	A[j][i] = temp;
        }
    }
}
```

*以上代码的功能是对矩阵进行转置，在进行遍历的时候，会进行检查，防止反转两次。但其实我们可以修改循环边界达到这样的效果*

```c
//在循环条件上就限制 j < i
for (int i = 1; i < n; ++i) {
  	for (int j = 0; j < i; ++j) {
         int temp = A[i][j];
         A[i][j] = A[j][i];
         A[j][i] = temp;
    }
}
```



#### 功能上的优化

> 内联的想法是通过将函数的调用替换为函数本身的主体来避免函数调用的开销。

Example:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img26.jpg)

*这种优化会在编译器内部实现，并不需要我们手动实现，所以说前面的写法并不会造成性能不佳，但这种优化的思想是我们要学习的。* 



> 尾部递归消除的想法是将作为函数最后一步的递归调用替换为分支，从而节省函数调用开销。

Example:

```c
// 消除前
void quicksort(int *A, int n) {
    if (n > 1){
        int r = partition(A, n);
        quicksort(A, r);
        quicksort(A + r + 1, n - r -1);
    }
}

// 消除后
void quicksort(int *A, int n){
    while(n > 1){
        int r = partition(A, n);
        quicksort(A, r);
        A += r + 1;
        n -= r + 1;
    }
}
```



> 粗化递归，是指在递归到一个比较小的case的时候，不继续进行递归，而是直接计算结果，通过这样的方式来减少函数调用。例如快排，我们可以在仅有几个数据的情况下直接进行某一种排序，这样即使其复杂度会达到$O(n^2)$，但是其数据量比较小并且减少了函数调用，还是可以使用的。

### Bits Hacks

#### 基础操作

> 部分基础知识

10进制&16进制&2进制:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img27.jpg)

位运算符：

```scss
& : AND
| : OR
^ : XOR(异或)
~ : NOT(取反)
<< : shift left
>> : shift right
```

Example:

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img28.jpg)



> 设置位

设置某二进制数x的第k位为1。我们需要一个在第k位为1的数做或运算，所以我们可以通过左移运算来实现：$y = x | (1 << 1)$执行过程如下图所示：

![](/Users/caixiongjiang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/143cc9c1b83000cdfc29000f3163b752/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/Image/3561688973248_.pic.jpg)



> 清零位

清零某二进制数x的第k位。我们可以通过与运算来实现清零，为了构造一个只有第k位为0的数，我们需要用到左移运算加非运算：$y = x \& ~(1 << k)$ 执行过程如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img30.jpg)



> 翻转位

翻转某二进制数x的第k位。我们可以通过异或运算来实现翻转，只需要在第k位和1异或就可以，自然而然的，我们需要构造第k位为1的数：$y = x \oplus(1 << k)$ 执行过程如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img31.jpg)



> 掩码操作

掩码操作是指，提取某二进制数x中的某一些位数，俗称mask，可以联想一下子网掩码： $y = x \& mask$ 执行过程如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img32.jpg)



> 提取一串数位

从二进制数x中提取某一些数位，我们可以通过掩码操作来提取出对应的数据，并通过移位操作移动shift位数放到指定的位置：$y = (x \& mask) >> shift$ 执行过程如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img33.jpg)



> 设置一串数位

设置二进制数x中的一些数位，和提取一串数位类似，我们通过掩码和移位操作。先通过与操作把这一些数位设置成0，然后通过移位操作和或操作进行设置： $x = (x \& \sim mask) | (y << shift)$执行过程如下图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img34.jpg)

为了安全性考虑，我们可以将y也进行一次掩码操作，这样保证只会设置我们需要的那些位数:

$x = (x \& \sim mask) | ((y << shift) \& mask)$



#### 高级操作

> 交换数字

交换变量x与y的值。我们可以通过temp作为中转，我们也可以通过位运算来实现。

其原理是来自于异或运算的特性。对于异或运算，连续两次异或同样的数，由于同样的数异或始终为0，而1和0异或上0都是他们自身，所以异或两次同样的数并不会修改当前数：$(x \oplus y)\oplus y = x$ 如图所示：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img35.jpg)

**我们可以使用该特性来进行函数交换：**
$$
x = x\oplus y\\
y = x\oplus y = (x\oplus y)\oplus y = x \oplus (y\oplus y) = x\\
x = x\oplus y = (x\oplus y)\oplus x = y \oplus (x\oplus x) = y
$$
变化过程如下：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img36.jpg)

代码如下：

```c
#include <stdio.h>

void swapNumbers(int *a, int *b) {
    *a = *a ^ *b;
    *b = *a ^ *b;
    *a = *a ^ *b;
}
```

*需要注意的是位运算的交换会比普通交换更慢一些，因为位运算交换存在顺序依赖性，不能并行计算。*



> 取最小值操作

min操作是一个经常用到的操作，取两个数中小的那个。一般来说，我们可以这样实现：

```c
if (x < y) {
  	r = x;
}else {
  	r = y;
}
//或者可以用三元表达式
r = (x < y) ? x : y;
```

在这样的代码中，如果不考虑编译器优化，就会出现一次跳转操作（被误判的分支清空了处理器管道），这个可能会影响程序性能。

使用位运算，我们可以使用这种方法：

```c
//c语言
r = y ^ ((x ^ y) & -(x - y));
```

为什么它可以起作用？

* 在c语言分别用整数1和0表示布尔值TRUE和FALSE。
* 如果`x < y`，那么`-(x - y) => -1`，在补码中使用`全1`表示，所以原式变为了`y ^ (x ^ y) => x`
* 如果`x >= y`，那么`-(x - y) => 0`，那么原式就变为了`y ^ 0 = y`



应用：取模运算

```c
r = (x + y) % n;
```

除法的开销大，可以换为三元表达式，

```c
z = x + y;
r = (z < n) ? z : z - n;
```

那么学习了位运算，我们可以转化为:

```c
z = x + y;
r = n ^ ((z ^ n) & -(z >= n));
```

**也就是说任何的不能预测的分支都可以变为位运算，这通常是编译器要做的事。**



> 快速计算2的次方数

对于2的k次方x，快速计算$lgx$也即k的值，可以通过都柏林序列的方式快速实现：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img37.jpg)



> 计算1的数量

我们可以通过如下的方式快速求得二进制数x中1的数量：

```c
for(r = 0; x != 0; ++r){
    x &= x - 1;
}
```

这是因为对于x而言，每次减1就以为着最右边的1会变减去，从而在与操作中变成0，如此重复即可获得1的数量。

该操作在1数量少的时候适用，当数量过大则不太适用。

我们也可以通过对照表的方式来实现，每次取出x的一部分进行查表运算，例如取出3位二进制，那么0-7对应的1的位数是已知的，加上该部分，继续取出下一个3位即可：

```c
static const int count[8] = {0,1,1,2,1,2,2,3};
for(int r = 0; x != 0; x >>=3){
    r += count[x & 0x08]
}
```

这个方式的快慢取决于内存操作的速度。

基于以上两种方式，我们可以用分治的方式来计算一个长串中1的数量。



### Assembly Language and Computer Architecture

#### 编译

程序从源代码编译到可执行文件需要四个步骤：

```scss
预处理 -> 编译 -> 汇编 -> 链接
```

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img48.jpg)

预处理会进行一些简单的代码替换等操作，比如将`#define`的内容进行替换，在这里我们不去讲述。

编译会将经过预处理过的代码进行编译，编译成汇编语言：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img49.jpg)

汇编语言是便于人去阅读而产生的，实际上机器并不能读懂汇编语言，所以我们需要将其转换成机器能读懂的二进制：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img50.jpg)

当我们完成了单个文件的编译过程，将单个文件代码编译成了二进制文件，但是在项目中，我们往往有多个项目文件需要一起编译，这时候就需要用链接器将代码链接到一起：

![](https://blog-1311257248.cos.ap-nanjing.myqcloud.com/imgs/%E9%AB%98%E6%80%A7%E8%83%BD%E8%AE%A1%E7%AE%97/img51.jpg)



> 为什么要学习汇编？

* 汇编能告诉我们编译器什么能干，什么不能干；

* 程序的bug可能来自于底层，比如开启过高级别的编译器优化时产生的bug；此外编译器也会有bug。
* 有时候我们不得不直接进行汇编级别的代码修改
* 逆向工程：当代码携带了debug信息后，我们可以用objdump进行反汇编。



#### X86-64指令集

