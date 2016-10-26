/*
**	Universidad de Buenos Aires
**	Facultad de Ingeniería
**	75.12 Análisis Numérico I
**	Trabajo Práctico I
**	Curso 3
**	13/10/2016
**
**	Merlo Leiva Nahuel
**	Padrón 92115
*/

#include <memory>
#include <limits.h>
#include <vector>
#include <map>
#include <fstream>
#include <string>

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

void Fill(int n, REAL* X, REAL value)
{
	for (int i = 0; i < n; i++)
	{
		X[i] = value;
	}
}

REAL InnerProduct(int n, const REAL* X, const REAL* Y)
{
	REAL result = 0;
	for (int i = 0; i < n; i++)
	{
		result += X[i] * Y[i];
	}
	return result;
}

void SolveLeastSquared(const std::map<REAL, REAL>& values, REAL* m, REAL* b)
{
	size_t size = values.size();
	REAL* phi_0 = new REAL[size];
	REAL* phi_1 = new REAL[size];
	REAL* f = new REAL[size];

	// Set phi_0
	Fill(size, phi_0, 1);

	// Set phi_1
	size_t pairIndex1 = 0;
	for each (auto pair in values)
	{
		phi_1[pairIndex1] = pair.first;
		pairIndex1++;
	}

	// Set f
	size_t pairIndex2 = 0;
	for each (auto pair in values)
	{
		f[pairIndex2] = pair.second;
		pairIndex2++;
	}

	// Compute SEL
	REAL phi_0_phi_0 = InnerProduct(size, phi_0, phi_0);
	REAL phi_0_phi_1 = InnerProduct(size, phi_0, phi_1);
	REAL phi_1_phi_1 = InnerProduct(size, phi_1, phi_1);
	REAL phi_1_phi_0 = InnerProduct(size, phi_1, phi_0);
	REAL phi_f_phi_0 = InnerProduct(size, f, phi_0);
	REAL phi_f_phi_1 = InnerProduct(size, f, phi_1);

	// Solve SEL
	REAL cramerDen = (phi_0_phi_0 * phi_1_phi_1 - phi_1_phi_0 * phi_0_phi_1);
	*b = (phi_f_phi_0 * phi_1_phi_1 - phi_1_phi_0 * phi_f_phi_1) / cramerDen;
	*m = (phi_0_phi_0 * phi_f_phi_1 - phi_f_phi_0 * phi_0_phi_1) / cramerDen;

	// Clean up
	delete[] phi_0;
	delete[] phi_1;
	delete[] f;
}

// Returns true if n is valid matrix dim.
bool ValidateN(int n)
{
	return !(n % submatrix_dim) &&
	n > submatrix_dim;
}

bool ComputeF_k(int n, int k, REAL* F_k, const REAL* X_k, const REAL* X_0, const REAL* X)
{
	REAL infiniteNormNum = 0;
	REAL infiniteNormDen = 0;
	for (int i = 0; i < n; i++)
	{
		REAL absDen = abs(X_0[i] - X[i]); // module
		if (absDen > infiniteNormDen) // max
		{
			infiniteNormDen = absDen;
		}

		REAL absNum = abs(X_k[i] - X[i]); // module
		if (absNum > infiniteNormNum) // max
		{
			infiniteNormNum = absNum;
		}
	}

	if (infiniteNormDen > 0)
	{
		*F_k = log10(infiniteNormNum / infiniteNormDen);
		return true;
	}
	else
	{
		printf_s("Invalid argument at ComputeF_k: X_k can't be zero.\n", n);
		return false;
	}
}

bool ComputeR_k(int n, REAL* R_k, const REAL* X_k, const REAL* X_k_minus_1)
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
	if (!ValidateN(n))
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
	if (!ValidateN(n))
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
	if (!ValidateN(n))
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

bool replace(std::string& str, const std::string& from, const std::string& to) {
	size_t start_pos = str.find(from);
	if (start_pos == std::string::npos)
		return false;
	str.replace(start_pos, from.length(), to);
	return true;
}

void SolveJacobiForN(int n, const REAL* B, const REAL* X, const REAL* X_0)
{
	REAL* x_in = new REAL[n];
	memset(x_in, 0, sizeof(REAL) * n);

	REAL* x_out = new REAL[n];
	memset(x_out, 0, sizeof(REAL) * n);

	std::map<REAL, REAL> ComputedF_k;

	REAL R_k = REAL_MAX;
	int stepCount = 0;
	while (R_k > rtol)
	{
		memcpy_s (x_in, sizeof(REAL) * n, x_out, sizeof(REAL) * n);
		SolveJacobi(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, x_in, x_out, B);
		ComputeR_k(n, &R_k, x_out, x_in);
		stepCount++;

		REAL F_k = 0;
		ComputeF_k(n, stepCount, &F_k, x_out, X_0, X);
		ComputedF_k.emplace(std::make_pair(stepCount, F_k));
	}

	// Estimate spectral radius
	REAL F_m = 0; // log(ro)
	REAL F_b = 0;
	SolveLeastSquared(ComputedF_k, &F_m, &F_b);
	REAL ro = pow(10, F_m);

	printf_s("Jacobi\n n=%d\t steps=%d\t rtol=%f\t ro=%f\n", n, stepCount, R_k, ro);
	for each (auto f_k in ComputedF_k)
	{
		printf_s("F*(%f)=%f\n", f_k.first, f_k.second);

		std::fstream file;
		std::string fileName;
		fileName.append("J");
		fileName.append(std::to_string(n));
		fileName.append(".txt");
		file.open(fileName, std::ios_base::out | std::ios_base::app);
		auto str = std::to_string(f_k.second);
		replace(str, ".", ",");
		str.append("\n");
		file.write(str.c_str(), str.size());
		file.close();
	}
	printf_s("\n");
	 
	delete[] x_in;
	delete[] x_out;
}

void SolveGaussSeidelForN(int n, const REAL* B, const REAL* X, const REAL* X_0, REAL* ro)
{
	REAL w = 1.0;

	REAL* x_in = new REAL[n];
	memset(x_in, 0, sizeof(REAL) * n);

	REAL* x_out = new REAL[n];
	memset(x_out, 0, sizeof(REAL) * n);

	REAL* r = new REAL[n];
	memset(r, 0, sizeof(REAL) * n);
	
	std::map<REAL, REAL> ComputedF_k;

	REAL R_k = REAL_MAX;
	int stepCount = 0;
	while (R_k > rtol)
	{
		memcpy_s(x_in, sizeof(REAL) * n, x_out, sizeof(REAL) * n);
		SolveSOR(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, r, x_out, B, w);
		ComputeR_k(n, &R_k, x_out, x_in);
		stepCount++;

		REAL F_k = 0;
		ComputeF_k(n, stepCount, &F_k, x_out, X_0, X);
		ComputedF_k.emplace(std::make_pair(stepCount, F_k));
	}

	// Estimate spectral radius
	REAL F_m = 0; // log(ro)
	REAL F_b = 0;
	SolveLeastSquared(ComputedF_k, &F_m, &F_b);
	*ro = pow(10, F_m);

	printf_s("Gauss Seidel\n n=%d\t steps=%d\t rtol=%f\t ro=%f\n", n, stepCount, R_k, *ro);
	for each (auto f_k in ComputedF_k)
	{
		printf_s("F*(%f)=%f\n", f_k.first, f_k.second);

		std::fstream file;
		std::string fileName;
		fileName.append("GS");
		fileName.append(std::to_string(n));
		fileName.append(".txt");
		file.open(fileName, std::ios_base::out | std::ios_base::app);
		auto str = std::to_string(f_k.second);
		replace(str, ".", ",");
		str.append("\n");
		file.write(str.c_str(), str.size());
		file.close();
	}
	printf_s("\n");

	delete[] x_in;
	delete[] x_out;
	delete[] r;
}

void SolveSORForN(int n, const REAL* B, const REAL* X, const REAL* X_0, REAL w)
{
	REAL* x_in = new REAL[n];
	memset(x_in, 0, sizeof(REAL) * n);

	REAL* x_out = new REAL[n];
	memset(x_out, 0, sizeof(REAL) * n);
	
	REAL* r = new REAL[n];
	memset(r, 0, sizeof(REAL) * n);

	std::map<REAL, REAL> ComputedF_k;

	REAL R_k = REAL_MAX;
	int stepCount = 0;
	while (R_k > rtol)
	{
		memcpy_s(x_in, sizeof(REAL) * n, x_out, sizeof(REAL) * n);
		SolveSOR(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, r, x_out, B, w);
		ComputeR_k(n, &R_k, x_out, x_in);
		stepCount++;

		REAL F_k = 0;
		ComputeF_k(n, stepCount, &F_k, x_out, X_0, X);
		ComputedF_k.emplace(std::make_pair(stepCount, F_k));
	}

	// Estimate spectral radius
	REAL F_m = 0; // log(ro)
	REAL F_b = 0;
	SolveLeastSquared(ComputedF_k, &F_m, &F_b);
	REAL ro = pow(10, F_m);

	printf_s("SOR\n n=%d\t steps=%d\t rtol=%f\t ro=%f\t w=%f\n", n, stepCount, R_k, ro, w);
	for each (auto f_k in ComputedF_k)
	{
		printf_s("F*(%f)=%f\n", f_k.first, f_k.second);

		std::fstream file;
		std::string fileName;
		fileName.append("SOR");
		fileName.append(std::to_string(n));
		fileName.append(".txt");
		file.open(fileName, std::ios_base::out | std::ios_base::app);
		auto str = std::to_string(f_k.second);
		replace(str, ".", ",");
		str.append("\n");
		file.write(str.c_str(), str.size());
		file.close();
	}
	printf_s("\n");

	delete[] x_in;
	delete[] x_out;
	delete[] r;
}

void SolveForN(int n)
{
	REAL* X = new REAL[n];
	Fill(n, X, 1);

	REAL* X_0 = new REAL[n];
	Fill(n, X_0, 0);

	REAL* B = new REAL[n];
	ComputeB(n, submatrix_B_1, submatrix_B_2, submatrix_B_3, submatrix_D_1, X, B);

	SolveJacobiForN(n, B, X, X_0);

	REAL gsRo = 0;
	SolveGaussSeidelForN(n, B, X, X_0, &gsRo);

	REAL optimalW = 2 / (1 + sqrt(1 - gsRo));
	SolveSORForN(n, B, X, X_0, optimalW);

	delete[] X;
	delete[] X_0;
	delete[] B;
}

int main()
{
	printf("Merlo Leiva Nahuel\n");
	printf("Padrón 92115\n");
	SolveForN(submatrix_dim * 2);
	SolveForN(submatrix_dim * 3);
	SolveForN(submatrix_dim * 4);
	SolveForN(submatrix_dim * 10);
	getchar();
	return 0;
}