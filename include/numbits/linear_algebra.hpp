#pragma once

#include "ndarray.hpp"
#include "operations.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <limits>
#include <functional>
#include <algorithm>

namespace numbits {

constexpr double TOL = 1e-10;

/**
 * @brief Performs matrix multiplication of two 2D ndarrays.
 *
 * @tparam T Numeric type (e.g., double, float)
 * @param a Left matrix of shape (m, n)
 * @param b Right matrix of shape (n, p)
 * @return ndarray<T> Resulting matrix of shape (m, p)
 * @throws std::runtime_error If input matrices are not 2D or shapes are incompatible
 */
template<typename T>
ndarray<T> matmul(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul requires 2D ndarrays");
    if (a.shape()[1] != b.shape()[0])
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");

    size_t m = a.shape()[0];
    size_t n = a.shape()[1];
    size_t p = b.shape()[1];
    ndarray<T> result(Shape{m, p});

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j) {
            T sum = T{0};
            for (size_t k = 0; k < n; ++k) sum += a.at({i,k}) * b.at({k,j});
            result.at({i,j}) = sum;
        }

    return result;
}

/**
 * @brief Computes the dot product for vectors or matrices.
 *
 * - For vectors: returns a 1-element ndarray containing the scalar dot product.
 * - For matrices: behaves like matmul.
 *
 * @tparam T Numeric type
 * @param a First array (1D or 2D)
 * @param b Second array (1D or 2D)
 * @return ndarray<T> Dot product result
 * @throws std::runtime_error If dimensions are incompatible or unsupported
 */
template<typename T>
ndarray<T> dot(const ndarray<T>& a, const ndarray<T>& b) {
    if (a.ndim() == 1 && b.ndim() == 1) {
        if (a.size() != b.size()) throw std::runtime_error("Vectors must have same size");
        T sum = T{0};
        for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
        return ndarray<T>({1}, {sum});
    }
    else if (a.ndim() == 2 && b.ndim() == 2) return matmul(a,b);
    else if (a.ndim() == 2 && b.ndim() == 1) {
        if (a.shape()[1] != b.size()) throw std::runtime_error("Incompatible shapes");
        ndarray<T> res({a.shape()[0]});
        for (size_t i = 0; i < a.shape()[0]; ++i) {
            T sum = 0;
            for (size_t j = 0; j < a.shape()[1]; ++j) sum += a.at({i,j})*b[j];
            res[i] = sum;
        }
        return res;
    } else throw std::runtime_error("Unsupported dimensions for dot");
}

/**
 * @brief Computes dot product of multiple matrices using optimal multiplication order.
 *
 * Uses dynamic programming to determine the least costly order of multiplication.
 *
 * @tparam T Numeric type
 * @param matrices Vector of 2D matrices to multiply
 * @return ndarray<T> Resulting matrix
 * @throws std::runtime_error If less than two matrices are provided or shapes are incompatible
 */
template<typename T>
ndarray<T> multi_dot(const std::vector<ndarray<T>>& matrices) {
    size_t n = matrices.size();
    if (n < 2) throw std::runtime_error("multi_dot requires at least two matrices");

    // Extract dimensions
    std::vector<size_t> dims(n+1);
    dims[0] = matrices[0].shape()[0];
    for (size_t i = 0; i < n; ++i) {
        if (matrices[i].ndim() != 2) throw std::runtime_error("All matrices must be 2D");
        if (i>0 && matrices[i].shape()[0]!=dims[i]) throw std::runtime_error("Incompatible shapes");
        dims[i+1] = matrices[i].shape()[1];
    }

    // DP tables
    std::vector<std::vector<size_t>> m(n,std::vector<size_t>(n,0));
    std::vector<std::vector<size_t>> s(n,std::vector<size_t>(n,0));

    for (size_t l=2;l<=n;++l) {
        for (size_t i=0;i<=n-l;++i) {
            size_t j=i+l-1;
            m[i][j] = std::numeric_limits<size_t>::max();
            for (size_t k=i;k<j;++k) {
                size_t q = m[i][k]+m[k+1][j]+dims[i]*dims[k+1]*dims[j+1];
                if (q<m[i][j]) { m[i][j]=q; s[i][j]=k; }
            }
        }
    }

    // Iterative computation
    std::function<ndarray<T>(size_t,size_t)> compute = [&](size_t i, size_t j) -> ndarray<T> {
        if (i==j) return matrices[i];
        ndarray<T> X = compute(i,s[i][j]);
        ndarray<T> Y = compute(s[i][j]+1,j);
        return matmul(X,Y);
    };
    return compute(0,n-1);
}

/**
 * @brief Raises a square matrix to an integer power.
 *
 * Uses exponentiation by squaring for efficiency. Supports negative powers by computing the inverse.
 *
 * @tparam T Numeric type
 * @param A Square matrix
 * @param n Integer power
 * @return ndarray<T> Resulting matrix A^n
 * @throws std::runtime_error If matrix is not square
 */
template<typename T>
ndarray<T> matrix_power(const ndarray<T>& A, int n) {
    if (A.ndim()!=2 || A.shape()[0]!=A.shape()[1])
        throw std::runtime_error("matrix_power requires square matrix");
    size_t sz=A.shape()[0];
    ndarray<T> result(Shape{sz,sz});
    for (size_t i=0;i<sz;++i)
        for (size_t j=0;j<sz;++j) result.at({i,j})=(i==j)?1:0;

    if (n==0) return result;
    ndarray<T> base = (n>0)?A:inverse(A);
    int exp = std::abs(n);
    while(exp>0) {
        if(exp%2==1) result = matmul(result,base);
        base = matmul(base,base);
        exp/=2;
    }
    return result;
}

/**
 * @brief Transposes a 2D matrix.
 *
 * @tparam T Numeric type
 * @param arr Input 2D matrix
 * @return ndarray<T> Transposed matrix
 * @throws std::runtime_error If array is not 2D
 */
template<typename T>
ndarray<T> transpose(const ndarray<T>& arr) {
    if(arr.ndim()!=2) throw std::runtime_error("transpose only supports 2D");
    size_t m=arr.shape()[0], n=arr.shape()[1];
    ndarray<T> res(Shape{n,m});
    for(size_t i=0;i<m;++i)
        for(size_t j=0;j<n;++j) res.at({j,i})=arr.at({i,j});
    return res;
}

/**
 * @brief Computes the determinant of a square matrix recursively.
 *
 * Uses Laplace expansion along the first row.
 *
 * @tparam T Numeric type
 * @param arr Square matrix
 * @return T Determinant value
 * @throws std::runtime_error If matrix is not square
 */
template<typename T>
T determinant(const ndarray<T>& arr) {
    if(arr.ndim()!=2 || arr.shape()[0]!=arr.shape()[1])
        throw std::runtime_error("determinant requires square matrix");
    size_t n=arr.shape()[0];
    if(n==1) return arr.at({0,0});
    if(n==2) return arr.at({0,0})*arr.at({1,1})-arr.at({0,1})*arr.at({1,0});
    T det=0;
    for(size_t j=0;j<n;++j) {
        ndarray<T> sub(Shape{n-1,n-1});
        for(size_t i=1;i<n;++i) {
            size_t col_idx=0;
            for(size_t k=0;k<n;++k) {
                if(k!=j) { sub.at({i-1,col_idx})=arr.at({i,k}); col_idx++; }
            }
        }
        T sign = (j%2==0)?1:-1;
        det += sign*arr.at({0,j})*determinant(sub);
    }
    return det;
}

/**
 * @brief Computes the inverse of a square matrix using Gaussian elimination.
 *
 * @tparam T Numeric type
 * @param A Square matrix
 * @return ndarray<T> Inverse of A
 * @throws std::runtime_error If matrix is not square or singular
 */
template<typename T>
ndarray<T> inverse(const ndarray<T>& A) {
    if(A.ndim()!=2 || A.shape()[0]!=A.shape()[1])
        throw std::runtime_error("inverse requires square matrix");
    size_t n = A.shape()[0];
    // Copy A to mutable matrix
    ndarray<T> mat = A;
    ndarray<T> inv(Shape{n,n});
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j) inv.at({i,j})=(i==j)?1:0;

    // Gaussian elimination with partial pivoting
    for(size_t i=0;i<n;++i) {
        // Pivot
        size_t max_row=i;
        for(size_t k=i+1;k<n;++k) if(std::abs(mat.at({k,i}))>std::abs(mat.at({max_row,i}))) max_row=k;
        if(std::abs(mat.at({max_row,i}))<TOL) throw std::runtime_error("Matrix is singular");
        if(max_row!=i) {
            for(size_t j=0;j<n;++j) { std::swap(mat.at({i,j}),mat.at({max_row,j})); std::swap(inv.at({i,j}),inv.at({max_row,j})); }
        }
        T pivot=mat.at({i,i});
        for(size_t j=0;j<n;++j) { mat.at({i,j})/=pivot; inv.at({i,j})/=pivot; }
        for(size_t k=0;k<n;++k) {
            if(k==i) continue;
            T factor=mat.at({k,i});
            for(size_t j=0;j<n;++j) { mat.at({k,j})-=factor*mat.at({i,j}); inv.at({k,j})-=factor*inv.at({i,j}); }
        }
    }
    return inv;
}

/**
 * @brief Solves the least-squares problem using SVD.
 *
 * Computes x that minimizes ||Ax - b|| using the pseudoinverse.
 *
 * @tparam T Numeric type
 * @param A Coefficient matrix
 * @param b Right-hand side (vector or matrix)
 * @return ndarray<T> Solution vector or matrix
 * @throws std::runtime_error If dimensions are incompatible
 */
template<typename T>
ndarray<T> lstsq(const ndarray<T>& A, const ndarray<T>& b) {
    if(A.ndim()!=2) throw std::runtime_error("A must be 2D");
    if(b.ndim()!=1 && b.ndim()!=2) throw std::runtime_error("b must be 1D or 2D");
    if(A.shape()[0]!=b.shape()[0]) throw std::runtime_error("Row count mismatch");

    ndarray<T> U,Vt;
    ndarray<T> S;
    svd_full(A,U,S,Vt);
    size_t k=S.size();

    // Compute pseudoinverse: A^+ = V Σ^+ U^T
    ndarray<T> Sigma_pinv(Shape{Vt.shape()[0],U.shape()[0]});
    for(size_t i=0;i<k;++i) Sigma_pinv.at({i,i}) = (S[i]>TOL)?1/S[i]:0;
    ndarray<T> Ut = transpose(U);
    ndarray<T> tmp = matmul(Sigma_pinv, Ut);
    ndarray<T> x;
    if(b.ndim()==1) {
        ndarray<T> b_col({b.size(),1});
        for(size_t i=0;i<b.size();++i) b_col.at({i,0})=b[i];
        x = matmul(tmp,b_col);
        ndarray<T> res({x.shape()[0]});
        for(size_t i=0;i<x.shape()[0];++i) res[i]=x.at({i,0});
        return res;
    } else x = matmul(tmp,b);
    return x;
}

/**
 * @brief Computes full singular value decomposition (SVD) of a matrix.
 *
 * Decomposes A into U Σ V^T.
 *
 * @tparam T Numeric type
 * @param A Input matrix
 * @param U Output orthogonal matrix U
 * @param S Output singular values (vector)
 * @param Vt Output orthogonal matrix V^T
 */
template<typename T>
void svd_full(const ndarray<T>& A, ndarray<T>& U, ndarray<T>& S, ndarray<T>& Vt) {
    const size_t m=A.shape()[0], n=A.shape()[1];
    const int max_iter=100;
    size_t k = std::min(m,n);
    ndarray<T> V(Shape{n,n});
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j) V.at({i,j})=(i==j)?1:0;
    ndarray<T> At = transpose(A);
    ndarray<T> AtA = matmul(At,A);

    // Jacobi rotations
    for(int iter=0;iter<max_iter;++iter){
        bool conv=true;
        for(size_t p=0;p<n;++p) for(size_t q=p+1;q<n;++q){
            T app=AtA.at({p,p}), aqq=AtA.at({q,q}), apq=AtA.at({p,q});
            if(std::abs(apq)>TOL){
                conv=false;
                T phi=0.5*std::atan2(2*apq,aqq-app);
                T c=std::cos(phi), s=std::sin(phi);
                for(size_t k2=0;k2<n;++k2){
                    T apk=AtA.at({p,k2}), aqk=AtA.at({q,k2});
                    AtA.at({p,k2})=c*apk-s*aqk;
                    AtA.at({q,k2})=s*apk+c*aqk;
                }
                for(size_t k2=0;k2<n;++k2){
                    T akp=AtA.at({k2,p}), akq=AtA.at({k2,q});
                    AtA.at({k2,p})=c*akp-s*akq;
                    AtA.at({k2,q})=s*akp+c*akq;
                }
                for(size_t k2=0;k2<n;++k2){
                    T vkp=V.at({k2,p}), vkq=V.at({k2,q});
                    V.at({k2,p})=c*vkp-s*vkq;
                    V.at({k2,q})=s*vkp+c*vkq;
                }
            }
        }
        if(conv) break;
    }

    S = ndarray<T>({k});
    for(size_t i=0;i<k;++i) S[i]=std::sqrt(std::max(AtA.at({i,i}),T{0}));

    U=ndarray<T>(Shape{m,m});
    for(size_t i=0;i<m;++i) for(size_t j=0;j<m;++j) U.at({i,j})=0;

    for(size_t j=0;j<k;++j){
        T sigma=S[j];
        if(sigma>TOL){
            for(size_t i=0;i<m;++i){
                T sum=0;
                for(size_t l=0;l<n;++l) sum+=A.at({i,l})*V.at({l,j});
                U.at({i,j})=sum/sigma;
            }
        } else for(size_t i=0;i<m;++i) U.at({i,j})=0;
    }

    // Complete U via modified Gram-Schmidt
    for(size_t j=k;j<m;++j){
        for(size_t i=0;i<m;++i) U.at({i,j})=(i==j)?1:0;
        for(size_t l=0;l<j;++l){
            T dot=0;
            for(size_t i=0;i<m;++i) dot+=U.at({i,l})*U.at({i,j});
            for(size_t i=0;i<m;++i) U.at({i,j})-=dot*U.at({i,l});
        }
        T norm_val=0;
        for(size_t i=0;i<m;++i) norm_val+=U.at({i,j})*U.at({i,j});
        norm_val=std::sqrt(norm_val);
        if(norm_val>TOL) for(size_t i=0;i<m;++i) U.at({i,j})/=norm_val;
    }

    Vt = transpose(V);
}

/**
 * @brief Computes the trace of a square matrix.
 *
 * @tparam T Numeric type
 * @param arr Square matrix
 * @return T Sum of diagonal elements
 * @throws std::runtime_error If matrix is not square
 */
template<typename T> T trace(const ndarray<T>& arr){
    if(arr.ndim()!=2 || arr.shape()[0]!=arr.shape()[1]) throw std::runtime_error("trace requires square matrix");
    T sum=0; for(size_t i=0;i<arr.shape()[0];++i) sum+=arr.at({i,i});
    return sum;
}

/**
 * @brief Computes Frobenius norm (matrix) or Euclidean norm (vector).
 *
 * @tparam T Numeric type
 * @param arr Input array
 * @return T Norm value
 */
template<typename T> T norm(const ndarray<T>& arr){
    T sum=0; for(size_t i=0;i<arr.size();++i) sum+=arr[i]*arr[i];
    return std::sqrt(sum);
}

/**
 * @brief Computes the outer product of two vectors.
 *
 * @tparam T Numeric type
 * @param a First vector
 * @param b Second vector
 * @return ndarray<T> Resulting matrix of shape (a.size(), b.size())
 * @throws std::runtime_error If inputs are not 1D vectors
 */
template<typename T> ndarray<T> outer(const ndarray<T>& a,const ndarray<T>& b){
    if(a.ndim()!=1 || b.ndim()!=1) throw std::runtime_error("outer requires 1D vectors");
    ndarray<T> res(Shape{a.size(),b.size()});
    for(size_t i=0;i<a.size();++i) for(size_t j=0;j<b.size();++j) res.at({i,j})=a[i]*b[j];
    return res;
}

/**
 * @brief Flattens an n-dimensional array into 1D.
 *
 * @tparam T Numeric type
 * @param arr Input array
 * @return ndarray<T> Flattened 1D array
 */
template<typename T> ndarray<T> flatten(const ndarray<T>& arr){
    ndarray<T> res({arr.size()});
    std::copy(arr.begin(),arr.end(),res.begin());
    return res;
}

} // namespace numbits
