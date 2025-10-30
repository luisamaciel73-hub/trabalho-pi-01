import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# =============================================================================
# PARTE 1: FUNÇÕES OBRIGATÓRIAS DO TRABALHO
# =============================================================================

def carregar_imagem(caminho_imagem):
    """
    Carrega uma imagem dos formatos jpg, png ou tiff com canais RGB.
    
    Args:
        caminho_imagem: Caminho para o arquivo de imagem
        
    Returns:
        Imagem carregada em formato numpy array (RGB)
    """
    imagem = Image.open(caminho_imagem)
    
    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
    
    imagem_array = np.array(imagem)
    
    print(f"✓ Imagem carregada: {caminho_imagem}")
    print(f"  Dimensões: {imagem_array.shape}")
    print(f"  Tipo: {imagem_array.dtype}")
    
    return imagem_array


def rgb_para_cinza(imagem_rgb):
    """
    Transforma uma imagem RGB em uma matriz de tons de cinza.
    Usa a fórmula padrão: Gray = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        imagem_rgb: Array numpy com imagem RGB (altura, largura, 3)
        
    Returns:
        Matriz de tons de cinza (altura, largura) com valores de 0 a 255
    """
    r = imagem_rgb[:, :, 0]
    g = imagem_rgb[:, :, 1]
    b = imagem_rgb[:, :, 2]
    
    cinza = 0.299 * r + 0.587 * g + 0.114 * b
    cinza = np.clip(cinza, 0, 255).astype(np.uint8)
    
    print(f"✓ Conversão para tons de cinza concluída")
    print(f"  Valores: mínimo={cinza.min()}, máximo={cinza.max()}")
    
    return cinza


def calcular_histograma(matriz_cinza):
    """
    Calcula o histograma da matriz em tons de cinza.
    
    Args:
        matriz_cinza: Matriz de tons de cinza (valores de 0 a 255)
        
    Returns:
        Array com 256 posições contendo a frequência de cada tom de cinza
    """
    histograma = np.zeros(256, dtype=int)
    
    for valor in range(256):
        histograma[valor] = np.sum(matriz_cinza == valor)
    
    print(f"✓ Histograma calculado")
    print(f"  Total de pixels: {np.sum(histograma)}")
    
    return histograma


def calcular_limiar_otsu(histograma):
    """
    Escolhe automaticamente um limiar usando o método de Otsu.
    Este método encontra o limiar que maximiza a variância entre classes.
    
    Args:
        histograma: Array com frequências de cada intensidade (0-255)
        
    Returns:
        Valor do limiar ótimo (0-255)
    """
    total_pixels = np.sum(histograma)
    
    soma_total = 0
    for i in range(256):
        soma_total += i * histograma[i]
    
    soma_fundo = 0
    peso_fundo = 0
    variancia_maxima = 0
    limiar_otimo = 0
    
    for limiar in range(256):
        peso_fundo += histograma[limiar]
        
        if peso_fundo == 0:
            continue
            
        peso_objeto = total_pixels - peso_fundo
        
        if peso_objeto == 0:
            break
        
        soma_fundo += limiar * histograma[limiar]
        
        media_fundo = soma_fundo / peso_fundo
        media_objeto = (soma_total - soma_fundo) / peso_objeto
        
        variancia_entre = peso_fundo * peso_objeto * (media_fundo - media_objeto) ** 2
        
        if variancia_entre > variancia_maxima:
            variancia_maxima = variancia_entre
            limiar_otimo = limiar
    
    print(f"✓ Limiar calculado pelo método de Otsu: {limiar_otimo}")
    
    return limiar_otimo


def binarizar_imagem(matriz_cinza, limiar):
    """
    Transforma a matriz de tons de cinza em uma matriz binária.
    
    Args:
        matriz_cinza: Matriz de tons de cinza
        limiar: Valor do limiar para binarização
        
    Returns:
        Matriz binária (apenas valores 0 e 255)
    """
    matriz_binaria = np.where(matriz_cinza > limiar, 255, 0).astype(np.uint8)
    
    pixels_brancos = np.sum(matriz_binaria == 255)
    pixels_pretos = np.sum(matriz_binaria == 0)
    total = matriz_binaria.size
    
    print(f"✓ Binarização concluída")
    print(f"  Pixels brancos: {pixels_brancos} ({pixels_brancos/total*100:.1f}%)")
    print(f"  Pixels pretos: {pixels_pretos} ({pixels_pretos/total*100:.1f}%)")
    
    return matriz_binaria


def salvar_imagem(matriz, caminho_saida):
    """
    Salva a matriz resultante como imagem.
    
    Args:
        matriz: Matriz de pixels
        caminho_saida: Caminho onde a imagem será salva
    """
    imagem = Image.fromarray(matriz.astype(np.uint8))
    imagem.save(caminho_saida)
    print(f"✓ Imagem salva em: {caminho_saida}")


# =============================================================================
# PARTE 2: CRIAR IMAGEM DE TESTE
# =============================================================================

def criar_imagem_teste():
    """Cria uma imagem de teste para demonstração"""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 50, 550, 350], fill='lightgray', outline='black', width=3)
    draw.ellipse([100, 100, 250, 250], fill='darkgray', outline='black', width=2)
    draw.rectangle([300, 100, 500, 250], fill='gray', outline='black', width=2)
    draw.polygon([150, 300, 250, 200, 350, 300], fill='dimgray', outline='black', width=2)
    
    img.save('imagem_teste.png')
    print("✓ Imagem de teste criada: imagem_teste.png")
    return 'imagem_teste.png'


# =============================================================================
# PARTE 3: VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================

def visualizar_resultados(imagem_original, matriz_cinza, histograma, matriz_binaria, limiar):
    """Visualiza todos os passos do processamento"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(imagem_original)
    axes[0, 0].set_title('1. Imagem Original (RGB)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(matriz_cinza, cmap='gray')
    axes[0, 1].set_title('2. Tons de Cinza', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].bar(range(256), histograma, color='gray', width=1)
    axes[0, 2].axvline(x=limiar, color='red', linestyle='--', linewidth=2, 
                       label=f'Limiar = {limiar}')
    axes[0, 2].set_title('3. Histograma', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Intensidade')
    axes[0, 2].set_ylabel('Frequência')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].imshow(matriz_binaria, cmap='gray')
    axes[1, 0].set_title(f'4. Imagem Binarizada (Limiar={limiar})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(matriz_cinza, cmap='gray')
    axes[1, 1].set_title('Antes da Binarização', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(matriz_binaria, cmap='gray')
    axes[1, 2].set_title('Depois da Binarização', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('resultado_completo.png', dpi=150, bbox_inches='tight')
    print("✓ Visualização salva em: resultado_completo.png")
    plt.show()


# =============================================================================
# PARTE 4: FUNÇÃO PRINCIPAL
# =============================================================================

def processar_imagem_completo(caminho_entrada, caminho_saida='imagem_binarizada.png'):
    """
    Função principal que executa todos os passos do processamento.
    """
    print("\n" + "="*70)
    print("PROCESSAMENTO DE LIMIARIZAÇÃO POR EQUILÍBRIO DO HISTOGRAMA")
    print("Método de Otsu")
    print("="*70 + "\n")
    
    print("[PASSO 1] Carregando imagem RGB...")
    imagem_original = carregar_imagem(caminho_entrada)
    
    print("\n[PASSO 2] Convertendo para tons de cinza...")
    matriz_cinza = rgb_para_cinza(imagem_original)
    
    print("\n[PASSO 3] Calculando histograma...")
    histograma = calcular_histograma(matriz_cinza)
    
    print("\n[PASSO 4] Calculando limiar automático (Otsu)...")
    limiar = calcular_limiar_otsu(histograma)
    
    print("\n[PASSO 5] Binarizando imagem...")
    matriz_binaria = binarizar_imagem(matriz_cinza, limiar)
    
    print("\n[PASSO 6] Salvando resultado...")
    salvar_imagem(matriz_binaria, caminho_saida)
    
    print("\n[BÔNUS] Criando visualização completa...")
    visualizar_resultados(imagem_original, matriz_cinza, histograma, matriz_binaria, limiar)
    
    print("\n" + "="*70)
    print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print("\nArquivos gerados:")
    print(f"  • {caminho_saida} - Imagem binarizada")
    print(f"  • resultado_completo.png - Visualização de todos os passos")
    print("\n")
    
    return imagem_original, matriz_cinza, histograma, matriz_binaria, limiar


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRABALHO 01 DE PI - LIMIARIZAÇÃO DE IMAGENS")
    print("="*70)
    
    caminho_teste = criar_imagem_teste()
    
    processar_imagem_completo(caminho_teste, 'imagem_binarizada.png')
    
    print("\n" + "="*70)
    print("COMO USAR COM SUAS PRÓPRIAS IMAGENS:")
    print("="*70)
    print("\nPara processar outra imagem, use:")
    print("  processar_imagem_completo('sua_imagem.jpg', 'saida.png')")
    print("\nOu execute diretamente:")
    print("  python trabalho_pi.py")
    print("="*70 + "\n")

processar_imagem_completo('teste.jpg', 'resultado.png')