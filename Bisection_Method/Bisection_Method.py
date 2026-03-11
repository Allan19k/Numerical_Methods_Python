"""
Implementación del Método de Bisección para encontrar raíces de funciones
Soporta: funciones complejas, polinómicas, trigonométricas, exponenciales, etc.
El programa solicita al usuario:
    1. La función f(x) en formato de código Python (con operadores y funciones estándar)
    2. El intervalo [a, b] donde se busca la raíz
    3. La tolerancia para la aproximación de la raíz
El programa verifica la existencia de raíz en el intervalo, realiza las iteraciones del método de bisección, y muestra los resultados detallados, en una tabla de iteraciones

    Ademas se le agregaron algunas mejoras para hacerlo mas robusto y amigable:
    1. Validación de la función ingresada, con mensajes de error claros
    2. Gráfica de la función con la raíz encontrada marcada, para visualizar mejor el resultado
    3. Manejo de casos especiales, como raíces exactas en los límites del intervalo
    4. Guía detallada para escribir la función correctamente, con ejemplos de traducción de expresiones matemáticas a código Python
    5. Formato de salida mejorado, con mensajes claros y una tabla de iteraciones bien organizada
    6. Guardado de las iteraciones en un archivo CSV con un formato claro y organizado, incluyendo información general y una tabla de iteraciones
    7. Gráfica de la convergencia del método (error aproximado vs iteraciones) para analizar el comportamiento del método
"""

import numpy as np
import re
import csv
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from datetime import datetime
import os

def preprocessar_expresion(expr: str) -> str:
    """Preprocesa la expresión para agregar multiplicaciones implícitas."""
    functions = ['log10', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 
                 'arcsin', 'arccos', 'arctan', 'log', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'abs']
    
    expr = re.sub(r'(x)(\d+)(?![0-9])', r'\1**\2', expr)
    expr = re.sub(r'(\))(\d+)(?![0-9])', r'\1**\2', expr)
    
    for func in functions:
        expr = re.sub(rf'(\d)\s*({re.escape(func)})\s*\(', rf'\1*{func}(', expr, flags=re.IGNORECASE)
    for func in functions:
        expr = re.sub(rf'(x)\s*({re.escape(func)})\s*\(', rf'x*{func}(', expr, flags=re.IGNORECASE)
        
    expr = re.sub(r'(\d)([x])\b', r'\1*\2', expr)
    
    for func in functions:
        expr = re.sub(rf'(\))\s*({re.escape(func)})\s*\(', rf')*{func}(', expr, flags=re.IGNORECASE)
        
    expr = re.sub(r'(\))\s*\(', r'\1*(', expr)
    return expr

def parsear_funcion(expresion: str) -> Callable:
    expr = expresion.lower().strip()
    expr = preprocessar_expresion(expr)
    
    def funcion(x):
        try:
            namespace = {
                'x': x,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'exp': np.exp, 'log': np.log, 'log10': np.log10,
                'sqrt': np.sqrt, 'abs': np.abs, 'pi': np.pi, 'e': np.e,
            }
            # Se permite __import__ para evitar que python crashee cuando intente dar 
            # un warning por división entre cero (como log10(1) = 0).
            result = eval(expr, {"__builtins__": {"__import__": __import__}}, namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluando la función en x={x}: {e}")
            
    return funcion

def verificar_raiz(f: Callable, a: float, b: float) -> Tuple[bool, float, float]:
    fa = f(a)
    fb = f(b)
    # Si alguno de los extremos da exactamente 0, esa YA ES la raíz
    if fa == 0.0 or fb == 0.0:
        return True, fa, fb
        
    existe = (fa * fb) < 0
    return existe, fa, fb

def metodo_biseccion(f: Callable, a: float, b: float, tolerancia: float = 0.01, 
                     max_iteraciones: int = 100) -> dict:
    historial = {
        'iteraciones': [], 'aproximaciones': [],
        'errores': [], 'valores_funcion': []
    }
    
    existe_raiz, fa, fb = verificar_raiz(f, a, b)
    
    # Manejo de raíces en los propios límites
    if fa == 0.0:
        return {'exito': True, 'mensaje': f'Raíz exacta encontrada en el límite a={a}', 'raiz': a, 'iteraciones': 0, 'f_raiz': fa, 'historial': historial}
    if fb == 0.0:
        return {'exito': True, 'mensaje': f'Raíz exacta encontrada en el límite b={b}', 'raiz': b, 'iteraciones': 0, 'f_raiz': fb, 'historial': historial}
        
    if not existe_raiz:
        return {'exito': False, 'mensaje': f'No existe raíz en [{a}, {b}]. f({a})={fa:.6f}, f({b})={fb:.6f}', 'raiz': None, 'iteraciones': 0, 'historial': historial}
    
    iteracion = 0
    raiz_anterior = None
    
    while iteracion < max_iteraciones:
        iteracion += 1
        c = (a + b) / 2.0
        fc = f(c)
        
        historial['iteraciones'].append(iteracion)
        historial['aproximaciones'].append(c)
        historial['valores_funcion'].append(fc)
        
        if raiz_anterior is not None:
            error = abs(c - raiz_anterior)
            historial['errores'].append(error)
        else:
            historial['errores'].append(None)
            
        raiz_anterior = c
        ancho_intervalo = abs(b - a)
        
        # En caso de que pegue exactamente en la raíz a mitad de camino
        if fc == 0.0:
            return {'exito': True, 'mensaje': 'Raíz exacta encontrada en el punto medio', 'raiz': c, 'iteraciones': iteracion, 'f_raiz': fc, 'historial': historial}
            
        if raiz_anterior is not None and ancho_intervalo / 2 < tolerancia:
            return {'exito': True, 'mensaje': 'Raíz encontrada con éxito (tolerancia alcanzada)', 'raiz': c, 'iteraciones': iteracion, 'f_raiz': fc, 'historial': historial}
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return {'exito': True, 'mensaje': 'Alcanzado número máximo de iteraciones', 'raiz': raiz_anterior, 'iteraciones': iteracion, 'f_raiz': fc, 'historial': historial}

def graficar_funcion(f: Callable, a: float, b: float, raiz: float, expr_original: str):
    """Grafica la función, el intervalo y la raíz encontrada."""
    try:
        # Crear puntos para la gráfica
        x = np.linspace(a, b, 1000)
        y = []
        
        for xi in x:
            try:
                y.append(f(xi))
            except:
                y.append(np.nan)  # Ignorar valores donde la función no está definida
        
        y = np.array(y)
        
        # Crear figura
        plt.figure(figsize=(12, 7))
        
        # Graficar la función
        plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
        
        # Graficar el eje x (y=0)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Marcar el intervalo [a, b]
        plt.axvline(x=a, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Límite a = {a}')
        plt.axvline(x=b, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Límite b = {b}')
        
        # Marcar la raíz encontrada
        f_raiz = f(raiz)
        plt.plot(raiz, f_raiz, 'ro', markersize=12, label=f'Raíz encontrada ≈ {raiz:.6f}', zorder=5)
        
        # Añadir líneas para visualizar mejor la raíz
        plt.axvline(x=raiz, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        
        # Configurar la gráfica
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.title(f'Método de Bisección: f(x) = {expr_original}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Mostrar la gráfica
        plt.show()
        
    except Exception as e:
        print(f" Error al graficar la función: {e}")

def graficar_convergencia(resultados: dict):
    """Grafica la convergencia del método: error aproximado vs iteraciones."""
    try:
        historial = resultados['historial']
        iteraciones = historial['iteraciones']
        errores = historial['errores']
        
        # Filtrar errores válidos (no None)
        iteraciones_validas = []
        errores_validos = []
        
        for i, error in enumerate(errores):
            if error is not None:
                iteraciones_validas.append(iteraciones[i])
                errores_validos.append(error)
        
        if not errores_validos:
            print("No hay datos de error para graficar la convergencia.")
            return
        
        # Crear figura
        plt.figure(figsize=(12, 7))
        
        # Graficar la convergencia en escala logarítmica
        plt.semilogy(iteraciones_validas, errores_validos, 'b-o', linewidth=2, markersize=8, label='Error aproximado')
        
        # Configurar la gráfica
        plt.xlabel('Número de Iteración', fontsize=12, fontweight='bold')
        plt.ylabel('Error Aproximado |Xₙ₊₁ - Xₙ| (escala logarítmica)', fontsize=12, fontweight='bold')
        plt.title('Convergencia del Método de Bisección', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, which='both')
        
        # Añadir información adicional
        plt.tight_layout()
        
        # Mostrar la gráfica
        plt.show()
        
    except Exception as e:
        print(f"Error al graficar la convergencia: {e}")

def guardar_iteraciones_csv(resultados: dict, expr_original: str, a: float, b: float, tolerancia: float):
    """Guarda las iteraciones en un archivo CSV en la carpeta de Descargas."""
    try:
        # Crear nombre de archivo con fecha y hora
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"iteraciones_biseccion_{timestamp}.csv"
        
        # Guardar en la carpeta Descargas del usuario
        directorio = os.path.join(os.path.expanduser("~"), "Downloads")
        
        ruta_archivo = os.path.join(directorio, nombre_archivo)
        
        # Escribir el archivo CSV
        with open(ruta_archivo, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Encabezado con información general
            writer.writerow(["METODO DE BISECCION - ITERACIONES"])
            writer.writerow([])
            writer.writerow(["Informacion General"])
            writer.writerow(["Funcion", expr_original])
            writer.writerow(["Intervalo [a, b]", f"[{a}, {b}]"])
            writer.writerow(["Tolerancia", tolerancia])
            writer.writerow(["Raiz encontrada", resultados['raiz']])
            writer.writerow(["f(raiz)", resultados['f_raiz']])
            writer.writerow(["Total de iteraciones", resultados['iteraciones']])
            writer.writerow([])
            
            # Encabezado de la tabla de iteraciones
            writer.writerow(["Iteracion", "Aproximacion", "f(x)", "Error Aproximado"])
            
            # Datos de iteraciones
            historial = resultados['historial']
            for i in range(len(historial['iteraciones'])):
                iter_num = historial['iteraciones'][i]
                aprox = historial['aproximaciones'][i]
                fx = historial['valores_funcion'][i]
                error = historial['errores'][i]
                
                error_str = f"{error:.10f}" if error is not None else "N/A"
                writer.writerow([iter_num, f"{aprox:.10f}", f"{fx:.10f}", error_str])
        
        print(f"Iteraciones guardadas en: {ruta_archivo}")
        return ruta_archivo
        
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
        return None

def mostrar_resultados(resultados: dict):
    print("\n" + "="*70)
    print("RESULTADOS DEL MÉTODO DE BISECCION".center(70))
    print("="*70)
    
    if not resultados['exito'] and resultados['raiz'] is None:
        print(f" {resultados['mensaje']}")
        return
    
    print(f"\n✓ {resultados['mensaje']}")
    print(f"\nRaíz encontrada: {resultados['raiz']:.10f}")
    print(f"f(raíz) = {resultados['f_raiz']:.2e}")
    print(f"Total de iteraciones: {resultados['iteraciones']}")
    
    if resultados['iteraciones'] > 0:
        print("\n" + "-"*70)
        print("TABLA DE ITERACIONES")
        print("-"*70)
        print(f"{'Iter':<6} {'Aproximacion':<20} {'f(x)':<15} {'Error Aprox.':<15}")
        print("-"*70)
        
        historial = resultados['historial']
        for i in range(len(historial['iteraciones'])):
            iter_num = historial['iteraciones'][i]
            aprox = historial['aproximaciones'][i]
            fx = historial['valores_funcion'][i]
            error = historial['errores'][i]
            
            error_str = f"{error:.6f}" if error is not None else "N/A"
            print(f"{iter_num:<6} {aprox:<20.10f} {fx:<15.6f} {error_str:<15}")
        print("-"*70)

def main():
    print("\n" + "="*80)
    print("MÉTODO DE BISECCIÓN PARA ENCONTRAR RAÍCES".center(80))
    print("="*80)
    print("\nEste programa encuentra raíces de funciones usando el método de bisección.")
    print("También se graficará la función y la convergencia del método para visualizar mejor los resultados.")
    print("\nAl finalizar, las iteraciones se guardarán en un archivo CSV en su carpeta de Descargas para que pueda revisarlas")
    
    print("\nGUÍA PARA ESCRIBIR SU FUNCIÓN f(x):")
    print("")
    print("  1. Operadores: + (suma), - (resta), * (multiplicación), / (división)")
    print("  2. Potencias: Use ** (Ejemplo: x² se escribe x**2)")
    print("  3. Raíces: sqrt(x) para raíz cuadrada. Para otras raíces, use fracciones:")
    print("     -> Raíz cúbica de x = (x)**(1/3)")
    print("  4. Funciones: sin(x), cos(x), tan(x), exp(x) [para eˣ], log(x) [Ln], log10(x)")
    print("  5. Fracciones complejas: Agrupe TODO el numerador y TODO el denominador:")
    print("     -> (Todo lo de arriba) / (Todo lo de abajo)")
    
    print("\nEJEMPLOS DE TRADUCCIÓN (DE PAPEL A CÓDIGO):")
    print("  • Matemática: 0.1x² - x·log(x)")
    print("")
    print("    Código:     0.1*x**2 - x*log10(x)    (Use log() si es logaritmo natural)")
    print("")
    print("  • Matemática:   (x-3)⁴ * sen(x)")
    print("                -------------------")
    print("                  (x² + 3x + 4)⁸   ")
    print("")
    print("    Código:     ((x - 3)**4 * sin(x)) / (x**2 + 3*x + 4)**8")
    print("")
    print("  • Matemática: (x²+5x-4) ∛(x²+2) eˣ")
    print("                -----------------------")
    print("                    log₁₀(x)(x+2)      ")
    print("")
    print("    Código:     ((x**2 + 5*x - 4) * (x**2 + 2)**(1/3) * exp(x)) / (log10(x) * (x + 2))")
    print("")
    print("  • Matemática: (x²+5x-4)² √(x²+2) eˣ")
    print("                -----------------------")
    print("                    log₁₀(x)(x+2)      ")
    print("")
    print("    Código:     ((x**2 + 5*x - 4)**2 * sqrt(x**2 + 2) * exp(x)) / (log10(x) * (x + 2))")
    print("="*90)
    
    while True:
        try:
            expr_funciones = input("\nFunción f(x) = ").strip()
            if not expr_funciones:
                continue
            
            f = parsear_funcion(expr_funciones)
            
            # Prueba rápida en x=1.5 (EVITAMOS 1.0 PARA NO DIVIDIR ENTRE log10(1)=0)
            f(1.5) 
            break
        except Exception as e:
            print(f"Error en la sintaxis de la función: {e}")
            print("Intente nuevamente con una expresión válida.\n")
            
    print("\nIngrese el intervalo [a, b]:")
    while True:
        try:
            a = float(input("  a = "))
            b = float(input("  b = "))
            if a >= b:
                print("Error: a debe ser menor que b")
                continue
            break
        except ValueError:
            print("Error: ingrese valores numéricos válidos")
            
    print("\nIngrese la tolerancia (Tol):")
    while True:
        try:
            tolerancia = float(input("  Tolerancia = "))
            if tolerancia <= 0:
                print("Error: la tolerancia debe ser mayor a 0")
                continue
            break
        except ValueError:
            print("Error: ingrese un valor numérico válido")
            
    print("\n" + "-"*70)
    print("Verificando condición de existencia de raíz...")
    existe, fa, fb = verificar_raiz(f, a, b)
    print(f"f({a}) = {fa:.6f}")
    print(f"f({b}) = {fb:.6f}")
    
    if existe:
        if fa == 0.0 or fb == 0.0:
            print(f"¡Uno de los límites ya es una raíz exacta!")
        else:
            print(f"Existe raíz en [{a}, {b}] (cambio de signo detectado)")
    else:
        print(f"NO existe raíz en [{a}, {b}] (no hay cambio de signo)")
        print("\nPor favor, ingresa un intervalo válido donde exista una raíz.")
        return
        
    print(f"\nCalculando raíz (tolerancia = {tolerancia})...")
    resultados = metodo_biseccion(f, a, b, tolerancia=tolerancia)
    
    mostrar_resultados(resultados)
    
    # Guardar iteraciones en CSV
    if resultados['iteraciones'] > 0:
        guardar_iteraciones_csv(resultados, expr_funciones, a, b, tolerancia)
    
    # Graficar la función si se encontró la raíz
    if resultados['exito'] and resultados['raiz'] is not None:
        print("\nGenerando gráficas...")
        graficar_funcion(f, a, b, resultados['raiz'], expr_funciones)
        
        # Graficar convergencia si hay múltiples iteraciones
        if resultados['iteraciones'] > 1:
            graficar_convergencia(resultados)
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()