
# 点到曲面距离计算

设高维空间体表达式为：

$$
f(x,y,z)=K
$$

设$K=k_0$为定值，对应该空间体在$K$方向上的截面，即：

$$
f(x,y,z)=k_0
$$

同时可以求得：

$$
z = g(x,y)
$$

对(2)两侧求导，有：

$$
\frac{\partial f}{\partial x}{\rm d}x + \frac{\partial f}{\partial y}{\rm d}y +\frac{\partial f}{\partial z}{\rm d}z = 0
$$

那么该平面在点$(x_0, y_0, z_0)$的切向量为：

$$
d = \left[
\delta x,  
\delta y,  
\frac{\frac{\partial f}{\partial x}|_{x_0} \delta x + \frac{\partial f}{\partial y}|_{y_0} \delta y}
  {-\frac{\partial f}{\partial z} |_{z_0}}
\right]
$$

该向量具有两个自由度$\delta x$和$\delta y$，对应一个切平面，该切平面的表达式为：

$$
z = -\frac{\partial f / \partial x}{\partial f / \partial z}\bigg|_{x_0, z_0}x
  -\frac{\partial f / \partial y}{\partial f / \partial z}\bigg|_{y_0, z_0} y
$$

法向量为：

$$
\vec{n} = \left[
  \frac{\partial f / \partial x}{\partial f / \partial z}\bigg|_{x_0, z_0},
  \frac{\partial f / \partial y}{\partial f / \partial z}\bigg|_{y_0, z_0},
  1
\right]
$$

则任意点$p = (x_p,y_p,z_p)$到平面的最短距离需要满足：

$$
[x_p, y_p, z_p]-[x_0, y_0, z_0] = N \cdot \vec n
$$

即：

$$
[x_p -x_0, y_p - y_0, z_p - g(x_0, y_0)] = N \cdot \left[
    \frac{\partial f / \partial x}{\partial f / \partial z}\bigg|_{x_0, z_0},
    \frac{\partial f / \partial y}{\partial f / \partial z}\bigg|_{y_0, z_0},
    1,
    \right]
$$

需要求解以下方程组获得曲面上最近邻点$(x_0 ,y_0, z_0)$：

$$
\left\{
    \begin{aligned} \\
    \frac{x_p - x_0}{\frac{\partial f / \partial x }{\partial f/ \partial z}\bigg|_{x_0, z_0}} = z_p - g(x_0, y_0)  \\
    \frac{y_p - y_0}{\frac{\partial f / \partial y}{\partial f/ \partial z}\bigg|_{y_0, z_0}} = z_p - g(x_0, y_0) \\
    \end{aligned}
\right.
$$

对应距离为：

$$
dist = \|(x_p, y_p, z_p) - (x_0, y_0, z_0)\|
$$

拓展到更一般形式，设曲面方程为：

$$
f(x_0, x_1, x_2, ..., x_{n-1}, x_n) = 0
$$

易有：

$$
x_{n}=g(x_0, x_1, ..., x_{n-1})
$$

对于任意曲面外的点$p = (x_{p,0},x_{p,1}, ... x_{p,n})$，需要求解以下方程组以获得曲面上的最近点位置$(x_0 ,..., x_n)$：

$$
\left\{
    \begin{aligned} \\
    \quad &\frac{x_{p, 0} - x_0}{\frac{\partial f / \partial x_0}{\partial f/ \partial x_n}\bigg|_{x_0, x_n}} = x_{p, n} - g(x_0, x_1, ..., x_{n-1})  \\
    \quad &\quad... \\
    \quad &\frac{x_{p, n-1} - x_{n-1}}{\frac{\partial f / \partial x_{n-1}}{\partial f/ \partial x_n}\bigg|_{x_{n-1}, x_n}} = x_{p, n} - g(x_0, x_1, ..., x_{n-1})  \\
    \end{aligned}
\right.
$$

对应距离为：

$$
dist = \|(x_{p,0}, ..., x_{p,n}) - (x_0, ..., x_n)\|
$$
