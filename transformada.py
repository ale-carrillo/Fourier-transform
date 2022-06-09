# TRANSFORMADA DE FOURIER

# Carrillo Cervantes Ivette Alejandra
# Morales Ortega Carlos
# Sanchez Hernandez Marco Antonio

#___________________________________________________________________#

# Importamos la biblioteca numpy
# Da soporte para crear arreglos y matrices
import numpy as np

#___________________________________________________________________#

# Función para mostrar los datos de la matriz
def mostrarM(tam): # init
    n = np.arange(tam)
    k = n.reshape((tam,1))
    M  = k*n
    print("M:\n", M) 

# Función para generar datos aleatorios
def datosAleatorios(tam): # init, tamaño del arreglo
    datos = np.random.random(tam)
    return datos 

# Función para obtener los terminos del circulo unitario
def terminosCirculares(N): # init
    terms =  np.exp(-1j *2*np.pi * np.arange(N)/N)
    return terms

#___________________________________________________________________#

# Función para obtener la Transformada Discreta de Fourier
def transformadaD(datos): # Array de 1 dimension
    tam = datos.shape[0] # Tamaño del arreglo  
    n = np.arange(tam)
    k = n.reshape((tam,1))
    matriz = np.exp(-1j * 2*np.pi * k * n/tam)
    return np.dot(matriz,datos)

# Función para obtener la Transformada Rapida de Fourier
def transformadaR(datos): # Arreglo de datos de 1 dimensión
    tam = datos.shape[0] # Tamaño del arreglo

    assert tam % 2 == 0, "el tamaño de los datos: {} debe ser una potencia de 2".format(tam)

    if tam <= 2:
        return transformadaD(datos)
    else:
        data_even = transformadaR(datos[::2])
        data_odd = transformadaR(datos[1::2])
        terms = terminosCirculares(tam)
        return np.concatenate([data_even + terms[:tam//2] * data_odd, data_even + terms[tam//2:] * data_odd])
    
#___________________________________________________________________#

print("\n_______________________________________________________\n")
print("\t\tTransformada Rapida de Fourier")
elementos = int(input("\n¿De cuantos elementos quiere su matriz? "))

arregloP = datosAleatorios(elementos)
print("\nArreglo principal:\n", arregloP)

dt =  transformadaD(arregloP)
fdft = transformadaR(arregloP)
dtnp = np.fft.fft(arregloP)

print("\nTransformada Discreta:\n",fdft)

print(np.allclose(dt,dtnp),
    np.allclose(fdft,dtnp))

print("")
mostrarM(elementos)

print("\n_______________________________________________________\n")

