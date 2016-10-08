
#include <memory>
#include <limits.h>

// Precision type abstraction.
typedef double REAL; 
#define REAL_MAX std::numeric_limits<REAL>::max()

// Tridiagonal, symmetric matrix.
// Each coefficient in the diagonal has the same value.
struct SUBMATRIX_TYPE_1
{
	REAL m_diag;
	REAL m_upper_lower_diag;
};

// Diagonal matrix.
// Each coefficient in the diagonal has the same value.
struct SUBMATRIX_TYPE_2
{
	REAL m_diag;
};

SUBMATRIX_TYPE_1 submatrix_B_1 = { 2, -0.5 };
SUBMATRIX_TYPE_1 submatrix_B_2 = { 4, -1 };
SUBMATRIX_TYPE_1 submatrix_B_3 = { 3, -0.5 };
SUBMATRIX_TYPE_2 submatrix_D_1 = { -1 };

const int submatrix_dim = 3;
const double rtol = 0.001;

// Returns true if n is valid matrix dim.
bool validateN(int n)
{
	return !(n % submatrix_dim) &&
	n > submatrix_dim;
}

bool ComputeM_k(int n, REAL* M_k, REAL* X_k, REAL* X_0, REAL* X, int k)
{

}

bool ComputeR_k(int n, REAL* R_k, REAL* X_k, REAL* X_k_minus_1)
{
	REAL infiniteNormNum = 0;
	REAL infiniteNormDen = 0;
	for (int i = 0; i < n; i++)
	{
		REAL absDen = abs(X_k[i]); // module
		if (absDen > infiniteNormDen) // max
		{
			infiniteNormDen = absDen;
		}

		REAL absNum = abs(X_k[i] - X_k_minus_1[i]); // module
		if (absNum > infiniteNormNum) // max
		{
			infiniteNormNum = absNum;
		}
	}

	if (infiniteNormDen > 0)
	{
		*R_k = infiniteNormNum / infiniteNormDen;
		return true;
	}
	else
	{
		printf_s("Invalid argument at ComputeR_k: X_k can't be zero.\n", n);
		return false;
	}
}

void ComputePartialBFromSubmatrix(
	int i,
	const SUBMATRIX_TYPE_1& submatrix,
	const REAL* x,
	REAL* b)
{
	b[i] = submatrix.m_diag * x[i];
	if (!(i % submatrix_dim))
	{
		b[i] += submatrix.m_upper_lower_diag * x[i + 1];
	}
	else if (!((i + 1) % submatrix_dim))
	{
		b[i] += submatrix.m_upper_lower_diag * x[i - 1];
	}
	else
	{
		b[i] += submatrix.m_upper_lower_diag * x[i + 1];
		b[i] += submatrix.m_upper_lower_diag * x[i - 1];
	}
}

bool ComputeB(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	const REAL* x,
	REAL* b
	)
{
	if (!validateN(n))
	{
		printf_s("Invalid argument n=%d at ComputeB.\n", n);
		return false;
	}
	else
	{
		for (int i = 0; i < submatrix_dim; i++)
		{
			ComputePartialBFromSubmatrix(i, b_1, x, b);
			b[i] += d_1.m_diag * x[i + submatrix_dim];
		}

		for (int i = submatrix_dim; i < n - submatrix_dim; i++)
		{
			ComputePartialBFromSubmatrix(i, b_2, x, b);
			b[i] += d_1.m_diag * x[i + submatrix_dim];
			b[i] += d_1.m_diag * x[i - submatrix_dim];
		}

		for (int i = n - submatrix_dim; i < n; i++)
		{
			ComputePartialBFromSubmatrix(i, b_3, x, b);
			b[i] += d_1.m_diag * x[i - submatrix_dim];
		}
		return true;
	}
}

void SolvePartialJacobiFromSubmatrix(
	int i,
	const SUBMATRIX_TYPE_1& submatrix,
	const REAL* X_k,
	REAL* X_k_plus_1,
	const REAL* b)
{
	X_k_plus_1[i] = b[i];
	if (!(i % submatrix_dim))
	{
		X_k_plus_1[i] -= submatrix.m_upper_lower_diag * X_k[i + 1];
	}
	else if (!((i + 1) % submatrix_dim))
	{
		X_k_plus_1[i] -= submatrix.m_upper_lower_diag * X_k[i - 1];
	}
	else
	{
		X_k_plus_1[i] -= submatrix.m_upper_lower_diag * X_k[i + 1];
		X_k_plus_1[i] -= submatrix.m_upper_lower_diag * X_k[i - 1];
	}
}

bool SolveJacobi(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	const REAL* X_k,
	REAL* X_k_plus_1,
	const REAL* b
	)
{
	if (!validateN(n))
	{
		printf_s("Invalid argument n=%d at SolveJacobi.\n", n);
		return false;
	}
	else
	{
		for (int i = 0; i < submatrix_dim; i++)
		{
			SolvePartialJacobiFromSubmatrix(i, b_1, X_k, X_k_plus_1, b);
			X_k_plus_1[i] -= d_1.m_diag * X_k[i + submatrix_dim];
			X_k_plus_1[i] /= b_1.m_diag;
		}

		for (int i = submatrix_dim; i < n - submatrix_dim; i++)
		{
			SolvePartialJacobiFromSubmatrix(i, b_2, X_k, X_k_plus_1, b);
			X_k_plus_1[i] -= d_1.m_diag * X_k[i + submatrix_dim];
			X_k_plus_1[i] -= d_1.m_diag * X_k[i - submatrix_dim];
			X_k_plus_1[i] /= b_2.m_diag;
		}

		for (int i = n - submatrix_dim; i < n; i++)
		{
			SolvePartialJacobiFromSubmatrix(i, b_3, X_k, X_k_plus_1, b);
			X_k_plus_1[i] -= d_1.m_diag * X_k[i - submatrix_dim];
			X_k_plus_1[i] /= b_3.m_diag;
		}
		return true;
	}
}

void SolvePartialSORFromSubmatrix(
	int i,
	const SUBMATRIX_TYPE_1& submatrix,
	REAL* r,
	REAL* X_k,
	const REAL* b)
{
	r[i] = b[i];
	if (!(i % submatrix_dim))
	{
		r[i] -= submatrix.m_upper_lower_diag * X_k[i + 1];
	}
	else if (!((i + 1) % submatrix_dim))
	{
		r[i] -= submatrix.m_upper_lower_diag * X_k[i - 1];
	}
	else
	{
		r[i] -= submatrix.m_upper_lower_diag * X_k[i + 1];
		r[i] -= submatrix.m_upper_lower_diag * X_k[i - 1];
	}
}

bool SolveSOR(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	REAL* r,
	REAL* X_k,
	const REAL* b,
	REAL w
	)
{
	if (!validateN(n))
	{
		printf_s("Invalid argument n=%d at SolveJacobi.\n", n);
		return false;
	}
	else
	{
		for (int i = 0; i < submatrix_dim; i++)
		{
			SolvePartialSORFromSubmatrix(i, b_1, r, X_k, b);
			r[i] -= d_1.m_diag * X_k[i + submatrix_dim];
			r[i] /= b_1.m_diag;
			r[i] -= X_k[i];
			X_k[i] += w * r[i];
		}

		for (int i = submatrix_dim; i < n - submatrix_dim; i++)
		{
			SolvePartialSORFromSubmatrix(i, b_2, r, X_k, b);
			r[i] -= d_1.m_diag * X_k[i + submatrix_dim];
			r[i] -= d_1.m_diag * X_k[i - submatrix_dim];
			r[i] /= b_2.m_diag;
			r[i] -= X_k[i];
			X_k[i] += w * r[i];
		}

		for (int i = n - submatrix_dim; i < n; i++)
		{
			SolvePartialSORFromSubmatrix(i, b_3, r, X_k, b);
			r[i] -= d_1.m_diag * X_k[i - submatrix_dim];
			r[i] /= b_3.m_diag;
			r[i] -= X_k[i];
			X_k[i] += w * r[i];
		}
		return true;
	}
}

void SolveGaussSeidel(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	REAL* r,
	REAL* X_k,
	const REAL* b)
{
	SolveSOR(n, b_1, b_2, b_3, d_1, r, X_k, b, 1.0);
}


void solveForN(int n)
{
	REAL* x = new REAL[n];
	memset(x, 0, sizeof(REAL) * n);

	REAL* x_in = new REAL[n];
	memset(x_in, 0, sizeof(REAL) * n);

	REAL* r = new REAL[n];
	memset(r, 0, sizeof(REAL) * n);

	REAL* x_1 = new REAL[n];
	for (int i = 0; i < n; i++)
	{
		x_1[i] = 1;
	}

	REAL* b = new REAL[n];
	memset(b, 0, sizeof(REAL) * n);
	ComputeB(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, x_1, b);

	REAL R_k = REAL_MAX;
	int jacobiStepCount = 0;
	while (R_k > rtol)
	{
		memcpy_s(x_in, sizeof(REAL) * n, x, sizeof(REAL) * n);
		SolveJacobi(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, x_in, x, b);
		ComputeR_k(n, &R_k, x, x_in);
		jacobiStepCount++;
	}
	printf_s("JACOBI n=%d steps=%d rtol=%f:\n", n, jacobiStepCount, R_k);

	memset(x, 0, sizeof(REAL) * n);
	R_k = REAL_MAX;
	REAL w = 1;
	int sorStepCount = 0;
	while (R_k > rtol)
	{
		memcpy_s(x_in, sizeof(REAL) * n, x, sizeof(REAL) * n);
		SolveSOR(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, r, x, b, w);
		ComputeR_k(n, &R_k, x, x_in);
		sorStepCount++;
	}
	printf_s("SOR(w=%f) n=%d steps=%d rtol=%f:\n", w, n, sorStepCount, R_k);

	delete[] x;
	delete[] r;
	delete[] x_1;
	delete[] x_in;
	delete[] b;
}

int main()
{
	solveForN(submatrix_dim * 2);
	solveForN(submatrix_dim * 3);
	solveForN(submatrix_dim * 4);
	solveForN(submatrix_dim * 10);
	getchar();
	return 0;
}