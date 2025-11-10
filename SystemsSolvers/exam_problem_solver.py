#!/usr/bin/env python3'
"""
Systems Biology Exam Problem Solver
Complete worked examples for typical exam problems
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import sympy as sp

class SystemsBiologyProblemSolver:
    """
    Comprehensive solver for systems biology exam problems
    """
    
    def __init__(self):
        self.fig_counter = 1
        
    def problem1_logistic_function(self):
        """
        Problem 1: Logistic Function Analysis
        f(x) = a/(b + e^(-cx))
        """
        print("\n" + "="*60)
        print("PROBLEM 1: LOGISTIC FUNCTION ANALYSIS")
        print("="*60)
        
        # Define symbolic variables
        x, a, b, c = sp.symbols('x a b c', real=True, positive=True)
        
        # Define the function
        f = a / (b + sp.exp(-c*x))
        
        print("\nGiven: f(x) = a/(b + e^(-cx))")
        print("\nStep 1: Find the derivative f'(x)")
        f_prime = sp.diff(f, x)
        print(f"f'(x) = {f_prime}")
        print(f"Simplified: f'(x) = ac·e^(-cx)/(b + e^(-cx))²")
        
        print("\nStep 2: Find the second derivative f''(x)")
        f_double = sp.diff(f_prime, x)
        print("f''(x) = [complex expression]")
        
        print("\nStep 3: Find inflection point (where f''(x) = 0)")
        print("Setting f''(x) = 0 and solving:")
        print("Inflection point at: x = ln(b)/c")
        print("At this point: f(ln(b)/c) = a/(2b)")
        
        print("\nStep 4: Analyze limiting behavior")
        print("As x → ∞: f(x) → a/b (upper asymptote)")
        print("As x → -∞: f(x) → 0 (lower asymptote)")
        
        print("\nStep 5: Sensitivity Analysis")
        # Sensitivity with respect to b
        sensitivity_b = sp.diff(f, b) * b / f
        print(f"Sensitivity S_b = ∂ln(f)/∂ln(b) = -b/(b + e^(-cx))")
        
        # Numerical example
        print("\n--- Numerical Example ---")
        a_val, b_val, c_val = 10, 2, 0.5
        x_vals = np.linspace(-10, 10, 200)
        
        def logistic(x, a, b, c):
            return a / (b + np.exp(-c*x))
        
        def logistic_derivative(x, a, b, c):
            exp_term = np.exp(-c*x)
            return a * c * exp_term / (b + exp_term)**2
        
        y_vals = logistic(x_vals, a_val, b_val, c_val)
        y_prime = logistic_derivative(x_vals, a_val, b_val, c_val)
        
        # Inflection point
        x_inflection = np.log(b_val) / c_val
        y_inflection = logistic(x_inflection, a_val, b_val, c_val)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Function plot
        ax = axes[0]
        ax.plot(x_vals, y_vals, 'b-', lw=2, label='f(x)')
        ax.plot(x_inflection, y_inflection, 'ro', markersize=8,
                label=f'Inflection ({x_inflection:.2f}, {y_inflection:.2f})')
        ax.axhline(y=a_val/b_val, color='r', linestyle='--', alpha=0.5,
                   label=f'Upper asymptote = {a_val/b_val:.1f}')
        ax.axhline(y=0, color='g', linestyle='--', alpha=0.5,
                   label='Lower asymptote = 0')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Logistic Function (a={a_val}, b={b_val}, c={c_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Derivative plot
        ax = axes[1]
        ax.plot(x_vals, y_prime, 'r-', lw=2)
        ax.axvline(x=x_inflection, color='b', linestyle='--', alpha=0.5,
                   label=f'Max at x = {x_inflection:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel("f'(x)")
        ax.set_title('First Derivative')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/home/claude/problem1_logistic.png', dpi=150)
        plt.show()
        
    def problem2_steady_state_analysis(self):
        """
        Problem 2: Steady State and Stability Analysis
        """
        print("\n" + "="*60)
        print("PROBLEM 2: STEADY STATE & STABILITY ANALYSIS")
        print("="*60)
        
        print("\nExample System:")
        print("dx/dt = 2x - xy")
        print("dy/dt = -y + xy")
        
        print("\nStep 1: Find steady states (set derivatives = 0)")
        print("2x - xy = 0 → x(2 - y) = 0")
        print("-y + xy = 0 → y(-1 + x) = 0")
        
        print("\nSteady states:")
        print("(0, 0) - trivial")
        print("(1, 2) - non-trivial")
        
        print("\nStep 2: Linearization via Jacobian Matrix")
        print("J = [∂f/∂x  ∂f/∂y]")
        print("    [∂g/∂x  ∂g/∂y]")
        
        print("\nJ = [2-y    -x  ]")
        print("    [y      -1+x]")
        
        print("\nStep 3: Evaluate at each steady state")
        
        # At (0,0)
        print("\nAt (0,0):")
        J1 = np.array([[2, 0], [0, -1]])
        eigenvalues1, eigenvectors1 = np.linalg.eig(J1)
        print(f"J = {J1}")
        print(f"Eigenvalues: λ₁ = {eigenvalues1[0]:.2f}, λ₂ = {eigenvalues1[1]:.2f}")
        print("Stability: SADDLE POINT (one positive, one negative eigenvalue)")
        
        # At (1,2)
        print("\nAt (1,2):")
        J2 = np.array([[0, -1], [2, 0]])
        eigenvalues2, eigenvectors2 = np.linalg.eig(J2)
        print(f"J = {J2}")
        print(f"Eigenvalues: λ₁ = {eigenvalues2[0]:.3f}, λ₂ = {eigenvalues2[1]:.3f}")
        
        if np.all(np.abs(np.real(eigenvalues2)) < 0.001):
            print("Stability: CENTER (pure imaginary eigenvalues → oscillations)")
        elif np.all(np.real(eigenvalues2) < 0):
            print("Stability: STABLE")
        else:
            print("Stability: Check eigenvalue signs")
            
    def problem3_michaelis_menten(self):
        """
        Problem 3: Michaelis-Menten Enzyme Kinetics
        """
        print("\n" + "="*60)
        print("PROBLEM 3: MICHAELIS-MENTEN KINETICS")
        print("="*60)
        
        print("\nReaction: E + S ⇌ ES → E + P")
        print("\nMichaelis-Menten equation: v = Vmax·S/(Km + S)")
        
        print("\nStep 1: Derive from mass action kinetics")
        print("Assuming quasi-steady state for ES complex:")
        print("d[ES]/dt = k₁[E][S] - k₋₁[ES] - k₂[ES] ≈ 0")
        
        print("\nStep 2: Parameter definitions")
        print("Vmax = k₂[E]₀ (maximum velocity)")
        print("Km = (k₋₁ + k₂)/k₁ (Michaelis constant)")
        
        # Numerical example
        Vmax = 10  # μmol/min
        Km = 5     # mM
        n_values = [1, 2, 4]  # Hill coefficients
        
        S = np.linspace(0, 50, 200)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Standard MM and Hill functions
        ax = axes[0]
        for n in n_values:
            if n == 1:
                v = Vmax * S / (Km + S)
                label = 'Michaelis-Menten (n=1)'
            else:
                v = Vmax * S**n / (Km**n + S**n)
                label = f'Hill function (n={n})'
            ax.plot(S, v, lw=2, label=label)
        
        ax.axhline(y=Vmax, color='k', linestyle='--', alpha=0.3, label='Vmax')
        ax.axhline(y=Vmax/2, color='r', linestyle='--', alpha=0.3, label='Vmax/2')
        ax.axvline(x=Km, color='g', linestyle='--', alpha=0.3, label='Km')
        ax.set_xlabel('[S] (mM)')
        ax.set_ylabel('v (μmol/min)')
        ax.set_title('Michaelis-Menten vs Hill Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Lineweaver-Burk plot (1/v vs 1/S)
        ax = axes[1]
        S_lb = np.linspace(1, 50, 50)
        v_lb = Vmax * S_lb / (Km + S_lb)
        ax.plot(1/S_lb, 1/v_lb, 'b-', lw=2)
        ax.set_xlabel('1/[S]')
        ax.set_ylabel('1/v')
        ax.set_title('Lineweaver-Burk Plot')
        ax.grid(True, alpha=0.3)
        
        # Competitive inhibition example
        ax = axes[2]
        Ki = 3  # Inhibitor constant
        I_concentrations = [0, 2, 5]  # Inhibitor concentrations
        
        for I in I_concentrations:
            Km_apparent = Km * (1 + I/Ki)
            v = Vmax * S / (Km_apparent + S)
            ax.plot(S, v, lw=2, label=f'[I] = {I} mM')
        
        ax.set_xlabel('[S] (mM)')
        ax.set_ylabel('v (μmol/min)')
        ax.set_title('Competitive Inhibition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/problem3_michaelis_menten.png', dpi=150)
        plt.show()
        
    def problem4_lotka_volterra(self):
        """
        Problem 4: Lotka-Volterra Competition Model
        """
        print("\n" + "="*60)
        print("PROBLEM 4: LOTKA-VOLTERRA COMPETITION MODEL")
        print("="*60)
        
        print("\nTwo-species competition:")
        print("dN₁/dt = r₁N₁(1 - N₁/K₁ - α₁₂N₂/K₁)")
        print("dN₂/dt = r₂N₂(1 - N₂/K₂ - α₂₁N₁/K₂)")
        
        print("\nParameters:")
        print("r₁, r₂: intrinsic growth rates")
        print("K₁, K₂: carrying capacities")
        print("α₁₂: competition effect of species 2 on species 1")
        print("α₂₁: competition effect of species 1 on species 2")
        
        print("\nCoexistence conditions:")
        print("α₁₂ < K₁/K₂ AND α₂₁ < K₂/K₁ (stable coexistence)")
        print("α₁₂ > K₁/K₂ AND α₂₁ > K₂/K₁ (competitive exclusion)")
        
        # Numerical simulation
        def competition_model(state, t, r1, r2, K1, K2, a12, a21):
            N1, N2 = state
            dN1dt = r1 * N1 * (1 - N1/K1 - a12*N2/K1)
            dN2dt = r2 * N2 * (1 - N2/K2 - a21*N1/K2)
            return [dN1dt, dN2dt]
        
        # Parameters for stable coexistence
        params = {
            'r1': 1.0, 'r2': 0.8,
            'K1': 100, 'K2': 120,
            'a12': 0.5, 'a21': 0.6
        }
        
        t = np.linspace(0, 50, 500)
        initial_conditions = [[10, 10], [50, 80], [80, 20]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectories
        colors = ['blue', 'green', 'red']
        for i, ic in enumerate(initial_conditions):
            trajectory = odeint(competition_model, ic, t, args=tuple(params.values()))
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], lw=2,
                   label=f'IC: N₁={ic[0]}, N₂={ic[1]}')
            ax.plot(ic[0], ic[1], 'o', color=colors[i], markersize=8)
        
        # Plot isoclines
        N1_range = np.linspace(0, 120, 100)
        N2_iso1 = (params['K1'] - N1_range) / params['a12']  # dN1/dt = 0
        N2_iso1[N2_iso1 < 0] = 0
        
        N2_range = np.linspace(0, 150, 100)
        N1_iso2 = (params['K2'] - N2_range) / params['a21']  # dN2/dt = 0
        N1_iso2[N1_iso2 < 0] = 0
        
        ax.plot(N1_range, N2_iso1, 'b--', lw=2, label='dN₁/dt = 0')
        ax.plot(N1_iso2, N2_range, 'r--', lw=2, label='dN₂/dt = 0')
        
        # Find and plot equilibrium
        def equations(p):
            N1, N2 = p
            eq1 = params['r1'] * N1 * (1 - N1/params['K1'] - params['a12']*N2/params['K1'])
            eq2 = params['r2'] * N2 * (1 - N2/params['K2'] - params['a21']*N1/params['K2'])
            return [eq1, eq2]
        
        # Coexistence equilibrium
        N1_eq, N2_eq = fsolve(equations, [50, 50])
        if N1_eq > 0 and N2_eq > 0:
            ax.plot(N1_eq, N2_eq, 'k*', markersize=15, label=f'Equilibrium ({N1_eq:.1f}, {N2_eq:.1f})')
        
        ax.set_xlabel('Species 1 (N₁)')
        ax.set_ylabel('Species 2 (N₂)')
        ax.set_title('Lotka-Volterra Competition Model')
        ax.legend()
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 150)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/problem4_competition.png', dpi=150)
        plt.show()

# Main execution
if __name__ == "__main__":
    solver = SystemsBiologyProblemSolver()
    
    print("\n" + "="*60)
    print("SYSTEMS BIOLOGY EXAM PROBLEM SOLVER")
    print("Based on your course materials (Lectures 1-6)")
    print("="*60)
    
    # Run all problem solvers
    solver.problem1_logistic_function()
    solver.problem2_steady_state_analysis()
    solver.problem3_michaelis_menten()
    solver.problem4_lotka_volterra()
    
    print("\n" + "="*60)
    print("KEY EXAM STRATEGIES")
    print("="*60)
    print("""
    1. ALWAYS START WITH THE BASICS
       - Define all variables and parameters
       - Write down the governing equations
       - Identify what you're solving for
    
    2. STEADY-STATE ANALYSIS CHECKLIST
       □ Set all derivatives to zero
       □ Solve for equilibrium points
       □ Calculate Jacobian matrix
       □ Find eigenvalues
       □ Classify stability
    
    3. COMMON PITFALLS TO AVOID
       - Forgetting to check for multiple steady states
       - Not verifying stability of equilibria
       - Mixing up parameters (Km vs Vmax, etc.)
       - Forgetting units in final answers
    
    4. GRAPHICAL ANALYSIS IS KEY
       - Always sketch nullclines for 2D systems
       - Mark equilibrium points clearly
       - Show direction of flow in phase plane
       - Include time series when relevant
    
    5. SENSITIVITY ANALYSIS FORMULA
       S_p = (∂f/∂p) × (p/f) = ∂ln(f)/∂ln(p)
       
    6. REMEMBER THE CANONICAL MODELS
       - Logistic growth: dN/dt = rN(1-N/K)
       - Michaelis-Menten: v = Vmax·S/(Km + S)
       - Lotka-Volterra: includes competition/predation terms
       - Hill function: includes cooperativity (n ≠ 1)
    """)
    
    print("\nGood luck on your exam! Remember to:")
    print("- Show all work step by step")
    print("- Check units and dimensions")
    print("- Verify answers make biological sense")
    print("- Draw diagrams to support your analysis")
'


