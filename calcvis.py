import numpy as np
from sympy import symbols, sympify, lambdify, integrate, diff, latex, sin, cos, tan, sec, csc, cot, asin, acos, atan, sqrt, log, E, Abs, Expr, Piecewise
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
import matplotlib.pyplot as plt
import streamlit as st
import re


#to parse string input into a callable function:
def parse_function(expr: str)->callable:
    #replaces ^ with ** which is python for "to the power of":
    expr = expr.replace("^", "**")
    #calls a function below to replace |x| with abs(x):
    expr = replaceBarsWithAbs(expr)
    #create a symbol 'x':
    x = symbols('x')
    #dictionary to hold all the operations and their sympy versions:
    local_dict = {
        "sin": sin, 
        "cos": cos, 
        "tan": tan, 
        "sec": sec, 
        "csc": csc, 
        "cot": cot,
        "asin": asin,
        "acos": acos,
        "atan": atan,
        "sqrt": sqrt,
        "log": log,
        "ln": log,
        "E": E,
        "e": E,
        "Abs": Abs,
        "abs": Abs
    }
    #for implicit multiplication:
    transformations = standard_transformations + (implicit_multiplication_application,)

    try:
        symbolic_expr = parse_expr(expr, local_dict=local_dict, transformations=transformations)
        f = lambdify(x, symbolic_expr, modules = ["numpy"])
        return f, symbolic_expr
    except Exception as e:
        print(f"[Error] could not parse function: {e}")
        return None, None

def riemann_sum(f, a, b, n, method)->float:

    dx = abs((b-a)/n)
    
    if method == 'left':
        x = [a + i * dx for i in range(n)]
    elif method == 'right':
        x = [a + (i + 1) * dx for i in range(n)]
    elif method == 'midpoint':
        x = [a + (i + 0.5) * dx for i in range(n)]
    else:
        raise ValueError("Method must be 'left', 'right', or 'midpoint'")
    
    heights = [f(xi) for xi in x]
    area = sum(h * dx for h in heights)
    return area, x, heights

def trapezoidal_rule(f, a, b, n):
    dx = (b-a)/n
    x_vals = np.linspace(a, b, n+1)
    y_vals = f(x_vals)
    area = dx * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])/2
    return area

def compute_integral(expr, a, b)->float:
    x = symbols('x')
    try:
        i = integrate(expr, (x, a, b))
        return float(i)
    except Exception as e:
        print (f"[Error] Could not compute integral: {e}")
        return None
    
def compute_derivative(expr: Expr):
    x = symbols('x')
    
    try:
        ddx= diff(expr, x)
        if ddx.has(type(diff(x, x))):
            ddx = ddx.doit()
        f_prime = lambdify(x, ddx, modules=[{"sin": np.sin, "cos": np.cos, "Abs": np.abs}, "numpy"])
        return ddx, f_prime
    except Exception as e:
        print(f"[Error] Could not differentiate {e}")
        return None, None

def replaceBarsWithAbs(expr: str)->str:
    return re.sub(r'\|([^|]+)\|', r'Abs(\1)', expr)
def generateRiemannNotation(expr_str, a, b, n, method):
    x = symbols('x')
    dx = r"\Delta x"
    xi = {
        'left': r"x_i",
        'right': r"x_{i+1}",
        'midpoint': r"\frac{x_i+x_{i+1}}{2}"
    }.get(method, r"x_i")
    local_dict = {
        "sin": sin, 
        "cos": cos, 
        "tan": tan, 
        "sec": sec, 
        "csc": csc, 
        "cot": cot,
        "asin": asin,
        "acos": acos,
        "atan": atan,
        "sqrt": sqrt,
        "log": log,
        "ln": log,
        "E": E,
        "e": E,
        "Abs": Abs,
        "abs": Abs
    }
    transformations = standard_transformations + (implicit_multiplication_application,)
    
    # Fix ^ and |x|:
    expr_str = expr_str.replace("^", "**")
    expr_str = re.sub(r'\|([^|]+)\|', r'Abs(\1)', expr_str)

    parsed_expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)

    return (rf"\sum_{{i=0}}^{{{n-1}}} f({xi}) {dx}, \quad"
            rf"{dx} = \frac{{{latex(b)}-{latex(a)}}}{{{n}}}")
def plot_functions(f=None, f_prime=None, a=0, b=5, show_func=True, show_deriv=True, title="graph", y_limits=None, show_rects=False, rect_data=None, dx=None, method='left'):
    
    x_vals = np.linspace(a, b, 500)
    plt.figure(figsize=(10, 6))

    if f and show_func:
        try:
            y_vals = f(x_vals)
            y_vals = np.nan_to_num(y_vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
            plt.plot(x_vals, y_vals, label="f(x)", color="blue")
        except Exception as e:
            st.error(f"Error plotting f(x): {e}")
        
    if f_prime and show_deriv:
        try:
            y_prime = f_prime(x_vals)
            y_prime = np.nan_to_num(y_prime, nan=np.nan, posinf=np.nan, neginf=np.nan)
            plt.plot(x_vals, y_prime, label="f'(x)", linestyle='--', color="red")
        except Exception as e:
            st.error(f"Error plotting f'(x): {e}")
    if y_limits:
        plt.ylim(y_limits)
    if show_rects and rect_data and dx:
        rect_x, rect_heights = rect_data

        if method == "midpoint":
            align_mode = "center"
        else:
            align_mode = "edge"

        if method == "right":
            rect_x = [x-dx for x in rect_x]

        plt.bar(rect_x, rect_heights, width=dx, align=align_mode, alpha=0.4, edgecolor='black', color='green', label="Riemann rectangles")

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    st.pyplot(plt.gcf())
    plt.clf()


#-------------------------------------------------------------------------------
#                    Begin User Interface (Streamlit)
#-------------------------------------------------------------------------------

st.title("Function Wizard")

raw_input = st.text_input("Enter a function f(x):", value="x^3 + sin(3x)")
expr_input = raw_input.replace("^", "**").replace("ln", "log")
st.subheader("x-axis scale:")
a = st.number_input("Lower bound (a):", value=0.0)
b = st.number_input("Upper bound (b):", value=3.0)
st.subheader("y-axis scale:")
y_min = st.number_input("y-axis lower bound:", value=-10.0)
y_max = st.number_input("y-axis upper bound:", value=10.0)
y_limits = (y_min, y_max)
show_rects = st.checkbox("Show riemann rectangles", value=True)
n = st.slider("Number of subintervals (n):", min_value=1, max_value=500, value=50)
method = st.selectbox("Riemann Sum method", ["left", "right", "midpoint", "trapezoidal"])
view = st.radio("What to visualize?", ["function only", "derivative only", "both"])
show_riemann =st.checkbox("show riemann sum notation", value=True)
sign = 1 if b>=a else -1

if expr_input and b != a:
    x = symbols('x')
    f, sym_expr = parse_function(expr_input)
    deriv_expr, f_prime = compute_derivative(sym_expr)
    antideriv_expr = integrate(sym_expr, x)
    if sym_expr is not None:
        st.subheader("Function:")
        st.latex(f"f'(x) = {latex(deriv_expr)}")
        st.latex(r"\boldsymbol{f(x) = " + latex(sym_expr)+r"}")
        st.latex(f"F(x) = {latex(antideriv_expr)} + C")
    if method == "trapezoidal":
        approx_area = trapezoidal_rule(f, a, b, n)
        rect_x = rect_heights = None
    else:
        area, rect_x, rect_heights = riemann_sum(f, a, b, n, method)
        approx_area = area
    exact_area = compute_integral(sym_expr, a, b)

    st.subheader("Results")
    if show_riemann:
        st.subheader("Riemann Sum Area Approximation:")
        st.latex(generateRiemannNotation(raw_input, a, b, n, method))
    st.write(f"Approximate Area ({method}): **{sign*approx_area:.5f}**")
    st.write(f"Exact Integral: **{sign*exact_area:.5f}**")
    st.write(f"Error: **{abs(exact_area - approx_area):.5f}**")

    st.subheader(f"Visualization:")
    show_func = view in ["function only", "both"]
    show_deriv = view in ["derivative only", "both"]
    rect_data=None 
    dx=None
    if method != "trapezoidal":
        _, rect_x, rect_heights = riemann_sum(f, a, b, n, method)
        dx = abs((b-a)/n)
        rect_data = (rect_x, rect_heights)
    plot_functions(
        f=f, 
        f_prime=f_prime, 
        a=a, 
        b=b, 
        show_func=show_func, 
        show_deriv=show_deriv, 
        y_limits=y_limits,
        show_rects=show_rects,
        rect_data=rect_data, 
        dx=dx,
        method=method)
else:
    st.warning("b is equal to a!")