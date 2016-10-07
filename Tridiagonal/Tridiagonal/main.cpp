
#include <memory>

// Precision type abstraction.
typedef double REAL;

// Tridiagonal, symmetric matrix.
// Each coefficient in the diagonal has the same value.
typedef struct SUBMATRIX_TYPE_1
{
	REAL m_diag;
	REAL m_upper_lower_diag;
};

// Diagonal matrix.
// Each coefficient in the diagonal has the same value.
typedef struct SUBMATRIX_TYPE_2
{
	REAL m_diag;
};

SUBMATRIX_TYPE_1 submatrix_B_1 = { 2, -0.5 };
SUBMATRIX_TYPE_1 submatrix_B_2 = { 4, -1 };
SUBMATRIX_TYPE_1 submatrix_B_3 = { 3, -0.5 };
SUBMATRIX_TYPE_2 submatrix_D_1 = { -1 };

const int submatrix_dim = 3;

// Returns true if n is valid matrix dim.
bool validateN(int n)
{
	return !(n % submatrix_dim) &&
	n > submatrix_dim;
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

void ComputeB(
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
	}
}

void SolvePartialJacobiFromSubmatrix(
	int i,
	const SUBMATRIX_TYPE_1& submatrix,
	const REAL* x_in,
	REAL* x_out,
	const REAL* b)
{
	x_out[i] = b[i];
	if (!(i % submatrix_dim))
	{
		x_out[i] -= submatrix.m_upper_lower_diag * x_in[i + 1];
	}
	else if (!((i + 1) % submatrix_dim))
	{
		x_out[i] -= submatrix.m_upper_lower_diag * x_in[i - 1];
	}
	else
	{
		x_out[i] -= submatrix.m_upper_lower_diag * x_in[i + 1];
		x_out[i] -= submatrix.m_upper_lower_diag * x_in[i - 1];
	}
}

void SolveJacobi(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	const REAL* x_in,
	REAL* x_out,
	const REAL* b
	)
{
	if (!validateN(n))
	{
		printf_s("Invalid argument n=%d at SolveJacobi.\n", n);
	}
	else
	{
		for (int i = 0; i < submatrix_dim; i++)
		{
			SolvePartialJacobiFromSubmatrix(i, b_1, x_in, x_out, b);
			x_out[i] -= d_1.m_diag * x_in[i + submatrix_dim];
			x_out[i] /= b_1.m_diag;
		}

		for (int i = submatrix_dim; i < n - submatrix_dim; i++)
		{
			SolvePartialJacobiFromSubmatrix(i, b_2, x_in, x_out, b);
			x_out[i] -= d_1.m_diag * x_in[i + submatrix_dim];
			x_out[i] -= d_1.m_diag * x_in[i - submatrix_dim];
			x_out[i] /= b_2.m_diag;
		}

		for (int i = n - submatrix_dim; i < n; i++)
		{
			SolvePartialJacobiFromSubmatrix(i, b_3, x_in, x_out, b);
			x_out[i] -= d_1.m_diag * x_in[i - submatrix_dim];
			x_out[i] /= b_3.m_diag;
		}
	}
}

void SolvePartialSORFromSubmatrix(
	int i,
	const SUBMATRIX_TYPE_1& submatrix,
	REAL* r,
	REAL* x,
	const REAL* b)
{
	r[i] = b[i];
	if (!(i % submatrix_dim))
	{
		r[i] -= submatrix.m_upper_lower_diag * x[i + 1];
	}
	else if (!((i + 1) % submatrix_dim))
	{
		r[i] -= submatrix.m_upper_lower_diag * x[i - 1];
	}
	else
	{
		r[i] -= submatrix.m_upper_lower_diag * x[i + 1];
		r[i] -= submatrix.m_upper_lower_diag * x[i - 1];
	}
}

void SolveSOR(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	REAL* r,
	REAL* x,
	const REAL* b,
	REAL w
	)
{
	if (!validateN(n))
	{
		printf_s("Invalid argument n=%d at SolveJacobi.\n", n);
	}
	else
	{
		for (int i = 0; i < submatrix_dim; i++)
		{
			SolvePartialSORFromSubmatrix(i, b_1, r, x, b);
			r[i] -= d_1.m_diag * x[i + submatrix_dim];
			r[i] /= b_1.m_diag;
			r[i] -= x[i];
			x[i] += w * r[i];
		}

		for (int i = submatrix_dim; i < n - submatrix_dim; i++)
		{
			SolvePartialSORFromSubmatrix(i, b_2, r, x, b);
			r[i] -= d_1.m_diag * x[i + submatrix_dim];
			r[i] -= d_1.m_diag * x[i - submatrix_dim];
			r[i] /= b_2.m_diag;
			r[i] -= x[i];
			x[i] += w * r[i];
		}

		for (int i = n - submatrix_dim; i < n; i++)
		{
			SolvePartialSORFromSubmatrix(i, b_3, r, x, b);
			r[i] -= d_1.m_diag * x[i - submatrix_dim];
			r[i] /= b_3.m_diag;
			r[i] -= x[i];
			x[i] += w * r[i];
		}
	}
}

void SolveGaussSeidel(
	int n,
	const SUBMATRIX_TYPE_1& b_1,
	const SUBMATRIX_TYPE_1& b_2,
	const SUBMATRIX_TYPE_1& b_3,
	const SUBMATRIX_TYPE_2& d_1,
	REAL* r,
	REAL* x,
	const REAL* b)
{
	SolveSOR(n, b_1, b_2, b_3, d_1, r, x, b, 1.0);
}


void solveForN(int n)
{
	REAL* x = new REAL[n];
	memset(x, 0, sizeof(REAL) * n);
	
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
	printf_s("B:\n");
	for (int i = 0; i < n; i++)
	{
		printf_s("A_%d=%f\n", i, b[i]);
	}

	for (int i = 0; i < 10; i++)
	{
		REAL* x_in = new REAL[n];
		memcpy_s(x_in, sizeof(REAL) * n, x, sizeof(REAL) * n);
		SolveJacobi(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, x_in, x, b);
	}
	printf_s("JACOBI:\n");
	for (int i = 0; i < n; i++)
	{
		printf_s("x_%d=%f\n", i, x[i]);
	}

	memset(x, 0, sizeof(REAL) * n);
	REAL w = 1.4;
	for (int i = 0; i < 10; i++)
	{
		SolveSOR(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, r, x, b, w);
	}
	printf_s("SOR(w=%f):\n", w);
	for (int i = 0; i < n; i++)
	{
		printf_s("x_%d=%f\n", i, x[i]);
	}

	delete[] x;
	delete[] r;
	delete[] x_1;
	delete[] b;
}

int main()
{
	solveForN(submatrix_dim * 3);
	getchar();
	return 0;
}