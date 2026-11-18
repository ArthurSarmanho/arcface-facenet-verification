import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from deepface import DeepFace

try:
    from deepface.modules import verification
except Exception:
    verification = None


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Avaliação de Reconhecimento Facial",
    page_icon="🔍",
    layout="wide"
)

st.title("Avaliação Comparativa de Reconhecimento Facial")

st.write(
    "Experimento comparativo entre **ArcFace** e **FaceNet**, "
    "utilizando **RetinaFace** para detecção facial."
)

st.divider()


# =========================================================
# CONFIGURAÇÕES DO EXPERIMENTO
# =========================================================

DETECTOR = "retinaface"
DISTANCE_METRIC = "cosine"
ALIGN = True
ENFORCE_DETECTION = True

TOTAL_EXECUCOES = 512


# =========================================================
# DATASET
# =========================================================

PESSOAS = {

    "Barack Obama": {
        "pasta": "Barack_obama",
        "refs": [
            "ref_obama.jpg",
            "ref2_obama.jpg"
        ],
        "testes": [
            f"teste{i}_obama.jpg"
            for i in range(1, 9)
        ]
    },

    "Neymar Jr.": {
        "pasta": "Neymar_jr",
        "refs": [
            "ref_neymar.jpg",
            "ref2_neymar.jpg"
        ],
        "testes": [
            f"teste{i}_neymar.jpg"
            for i in range(1, 9)
        ]
    },

    "Scarlett Johansson": {
        "pasta": "Scarlett_johansson",
        "refs": [
            "ref_scarlett.jpg",
            "ref2_scarlett.jpg"
        ],
        "testes": [
            f"teste{i}_scarlett.jpg"
            for i in range(1, 9)
        ]
    },

    "Taylor Swift": {
        "pasta": "Taylor_swift",
        "refs": [
            "ref_taylor.jpg",
            "ref2_taylor.jpg"
        ],
        "testes": [
            f"teste{i}_taylor.jpg"
            for i in range(1, 9)
        ]
    }
}


MODELOS = [
    "ArcFace",
    "Facenet"
]


# =========================================================
# PASTAS / ARQUIVOS
# =========================================================

PASTA_RESULTADOS = os.path.join(
    "resultados",
    "experimento_final"
)

os.makedirs(
    PASTA_RESULTADOS,
    exist_ok=True
)


CACHE_DETECCAO = os.path.join(
    PASTA_RESULTADOS,
    "cache_deteccao_retinaface.pkl"
)

CACHE_ARCFACE = os.path.join(
    PASTA_RESULTADOS,
    "embeddings_arcface.pkl"
)

CACHE_FACENET = os.path.join(
    PASTA_RESULTADOS,
    "embeddings_facenet.pkl"
)

ARQUIVO_FACES = os.path.join(
    PASTA_RESULTADOS,
    "contagem_faces_retinaface.csv"
)

ARQUIVO_RESULTADOS = os.path.join(
    PASTA_RESULTADOS,
    "resultados_completos_retinaface.csv"
)


# =========================================================
# FUNÇÕES GERAIS
# =========================================================

def nome_modelo(modelo):

    if modelo == "Facenet":
        return "FaceNet"

    return modelo


def caminho_imagem(pessoa, arquivo):

    return os.path.join(
        "dataset",
        PESSOAS[pessoa]["pasta"],
        arquivo
    )


def assinatura_arquivo(caminho):

    stat = os.stat(caminho)

    return {
        "tamanho": stat.st_size,
        "modificado": stat.st_mtime_ns
    }


def carregar_pickle(caminho):

    if not os.path.exists(caminho):
        return {}

    try:

        with open(caminho, "rb") as arquivo:
            return pickle.load(arquivo)

    except Exception:
        return {}


def salvar_pickle(objeto, caminho):

    with open(caminho, "wb") as arquivo:
        pickle.dump(
            objeto,
            arquivo,
            protocol=pickle.HIGHEST_PROTOCOL
        )


def assinatura_valida(registro, caminho):

    if not registro:
        return False

    assinatura_salva = registro.get(
        "assinatura"
    )

    if assinatura_salva is None:
        return False

    try:

        return assinatura_salva == assinatura_arquivo(
            caminho
        )

    except Exception:
        return False


def classificar(
    esperado_mesma_pessoa,
    verificado
):

    if esperado_mesma_pessoa:

        if verificado:
            return "VP"

        return "FN"

    else:

        if verificado:
            return "FP"

        return "VN"


def cenario_faces(quantidade):

    if quantidade is None:
        return "Não determinado"

    if quantidade == 0:
        return "Nenhuma face"

    if quantidade == 1:
        return "Uma face"

    return "Múltiplas faces"


def mensagem_erro(erro):

    partes = []
    atual = erro
    visitados = set()

    while (
        atual is not None
        and id(atual) not in visitados
    ):

        visitados.add(
            id(atual)
        )

        partes.append(
            f"{type(atual).__name__}: {str(atual)}"
        )

        if atual.__cause__ is not None:

            atual = atual.__cause__

        elif atual.__context__ is not None:

            atual = atual.__context__

        else:

            break

    return " | ".join(partes)


def eh_falha_deteccao(erro):

    texto = mensagem_erro(
        erro
    )

    return (
        "FaceNotDetected" in texto
        or
        "Face could not be detected" in texto
    )


# =========================================================
# DISTÂNCIA / THRESHOLD
# =========================================================

def distancia_cosseno(a, b):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)

    if norma_a == 0 or norma_b == 0:

        raise ValueError(
            "Embedding com norma zero."
        )

    similaridade = np.dot(
        a,
        b
    ) / (
        norma_a * norma_b
    )

    distancia = 1.0 - similaridade

    return round(
        float(distancia),
        6
    )


def obter_threshold(modelo):

    # Primeiro tenta utilizar exatamente a função
    # interna da versão instalada do DeepFace.

    if verification is not None:

        try:

            return float(
                verification.find_threshold(
                    modelo,
                    DISTANCE_METRIC
                )
            )

        except Exception:
            pass

    # Fallback compatível com os thresholds
    # utilizados no ambiente deste experimento.

    fallback = {
        "ArcFace": 0.68,
        "Facenet": 0.40
    }

    return fallback[modelo]


# =========================================================
# LISTA DAS 40 IMAGENS
# =========================================================

def listar_imagens():

    imagens = []

    for pessoa, dados in PESSOAS.items():

        for arquivo in dados["refs"]:

            imagens.append({
                "pessoa": pessoa,
                "arquivo": arquivo,
                "tipo": "Referência",
                "caminho": caminho_imagem(
                    pessoa,
                    arquivo
                )
            })

        for arquivo in dados["testes"]:

            imagens.append({
                "pessoa": pessoa,
                "arquivo": arquivo,
                "tipo": "Teste",
                "caminho": caminho_imagem(
                    pessoa,
                    arquivo
                )
            })

    return imagens


# =========================================================
# ETAPA 1
# DETECÇÃO COM RETINAFACE UMA ÚNICA VEZ
# =========================================================

def detectar_faces_dataset(
    barra=None,
    mensagem=None
):

    imagens = listar_imagens()

    cache = carregar_pickle(
        CACHE_DETECCAO
    )

    total = len(imagens)

    for indice, item in enumerate(
        imagens,
        start=1
    ):

        caminho = item["caminho"]

        registro_existente = cache.get(
            caminho
        )

        # Se a imagem já foi processada e não mudou,
        # utiliza o cache.

        if assinatura_valida(
            registro_existente,
            caminho
        ):

            if mensagem is not None:

                mensagem.write(
                    f"Detecção {indice}/{total} — "
                    f"cache: **{item['arquivo']}**"
                )

            if barra is not None:

                barra.progress(
                    indice / total
                )

            continue

        if mensagem is not None:

            mensagem.write(
                f"Detecção {indice}/{total} — "
                f"RetinaFace: **{item['arquivo']}**"
            )

        try:

            faces_detectadas = DeepFace.extract_faces(
                img_path=caminho,
                detector_backend=DETECTOR,
                enforce_detection=ENFORCE_DETECTION,
                align=ALIGN
            )

            faces_processadas = []

            for numero, face_obj in enumerate(
                faces_detectadas,
                start=1
            ):

                face_rgb = np.asarray(
                    face_obj["face"]
                )

                # O DeepFace normalmente retorna
                # a face normalizada entre 0 e 1.
                if face_rgb.max() <= 1.5:

                    face_rgb = (
                        face_rgb * 255.0
                    )

                face_rgb = np.clip(
                    face_rgb,
                    0,
                    255
                ).astype(
                    np.uint8
                )

                # represent() recebe arrays em BGR.
                face_bgr = face_rgb[
                    :,
                    :,
                    ::-1
                ].copy()

                area = face_obj.get(
                    "facial_area",
                    {}
                )

                faces_processadas.append({

                    "indice_face":
                        numero,

                    "face_bgr":
                        face_bgr,

                    "area":
                        {
                            "x": area.get("x"),
                            "y": area.get("y"),
                            "w": area.get("w"),
                            "h": area.get("h")
                        },

                    "confianca":
                        face_obj.get(
                            "confidence"
                        )
                })

            cache[caminho] = {

                "assinatura":
                    assinatura_arquivo(
                        caminho
                    ),

                "status":
                    "OK",

                "erro":
                    "",

                "faces":
                    faces_processadas
            }

        except Exception as erro:

            status = (
                "Falha de detecção"
                if eh_falha_deteccao(erro)
                else
                "Erro de processamento"
            )

            cache[caminho] = {

                "assinatura":
                    assinatura_arquivo(
                        caminho
                    ),

                "status":
                    status,

                "erro":
                    mensagem_erro(
                        erro
                    ),

                "faces":
                    []
            }

        # Salva depois de CADA imagem.
        # Se o programa for interrompido,
        # não perde o que já foi feito.

        salvar_pickle(
            cache,
            CACHE_DETECCAO
        )

        if barra is not None:

            barra.progress(
                indice / total
            )

    return cache


# =========================================================
# CRIA CSV DE CONTAGEM DE FACES
# =========================================================

def criar_csv_faces(
    cache_deteccao
):

    registros = []

    for item in listar_imagens():

        caminho = item["caminho"]

        dados = cache_deteccao.get(
            caminho,
            {}
        )

        faces = dados.get(
            "faces",
            []
        )

        quantidade = (
            len(faces)
            if dados.get("status") == "OK"
            else None
        )

        registros.append({

            "Pessoa":
                item["pessoa"],

            "Arquivo":
                item["arquivo"],

            "Tipo_imagem":
                item["tipo"],

            "Quantidade_faces":
                quantidade,

            "Cenario":
                cenario_faces(
                    quantidade
                ),

            "Detector":
                DETECTOR,

            "Status":
                dados.get(
                    "status",
                    "Não processado"
                ),

            "Erro":
                dados.get(
                    "erro",
                    ""
                )
        })

    df = pd.DataFrame(
        registros
    )

    df.to_csv(
        ARQUIVO_FACES,
        index=False,
        encoding="utf-8-sig"
    )

    return df


# =========================================================
# ETAPA 2
# GERA EMBEDDINGS SEM RODAR RETINAFACE NOVAMENTE
# =========================================================

def caminho_cache_modelo(modelo):

    if modelo == "ArcFace":
        return CACHE_ARCFACE

    return CACHE_FACENET


def gerar_embeddings_modelo(
    modelo,
    cache_deteccao,
    barra=None,
    mensagem=None
):

    caminho_cache = caminho_cache_modelo(
        modelo
    )

    cache_embeddings = carregar_pickle(
        caminho_cache
    )

    imagens = listar_imagens()

    total = len(imagens)

    for indice, item in enumerate(
        imagens,
        start=1
    ):

        caminho = item["caminho"]

        deteccao = cache_deteccao.get(
            caminho,
            {}
        )

        assinatura = assinatura_arquivo(
            caminho
        )

        existente = cache_embeddings.get(
            caminho
        )

        if (
            existente
            and existente.get(
                "assinatura"
            ) == assinatura
            and existente.get(
                "modelo"
            ) == modelo
        ):

            if mensagem is not None:

                mensagem.write(
                    f"{nome_modelo(modelo)} "
                    f"{indice}/{total} — cache: "
                    f"**{item['arquivo']}**"
                )

            if barra is not None:

                barra.progress(
                    indice / total
                )

            continue

        if mensagem is not None:

            mensagem.write(
                f"{nome_modelo(modelo)} "
                f"{indice}/{total} — "
                f"gerando embeddings: "
                f"**{item['arquivo']}**"
            )

        if deteccao.get(
            "status"
        ) != "OK":

            cache_embeddings[caminho] = {

                "assinatura":
                    assinatura,

                "modelo":
                    modelo,

                "status":
                    deteccao.get(
                        "status",
                        "Erro de detecção"
                    ),

                "erro":
                    deteccao.get(
                        "erro",
                        ""
                    ),

                "faces":
                    []
            }

            salvar_pickle(
                cache_embeddings,
                caminho_cache
            )

            if barra is not None:

                barra.progress(
                    indice / total
                )

            continue

        try:

            faces_com_embeddings = []

            for face in deteccao[
                "faces"
            ]:

                representacoes = DeepFace.represent(
                    img_path=face["face_bgr"],
                    model_name=modelo,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                    normalization="base"
                )

                embedding = representacoes[
                    0
                ][
                    "embedding"
                ]

                faces_com_embeddings.append({

                    "indice_face":
                        face[
                            "indice_face"
                        ],

                    "embedding":
                        embedding,

                    "area":
                        face[
                            "area"
                        ],

                    "confianca":
                        face[
                            "confianca"
                        ]
                })

            cache_embeddings[caminho] = {

                "assinatura":
                    assinatura,

                "modelo":
                    modelo,

                "status":
                    "OK",

                "erro":
                    "",

                "faces":
                    faces_com_embeddings
            }

        except Exception as erro:

            cache_embeddings[caminho] = {

                "assinatura":
                    assinatura,

                "modelo":
                    modelo,

                "status":
                    "Erro de embedding",

                "erro":
                    mensagem_erro(
                        erro
                    ),

                "faces":
                    []
            }

        # Salva após cada imagem.

        salvar_pickle(
            cache_embeddings,
            caminho_cache
        )

        if barra is not None:

            barra.progress(
                indice / total
            )

    return cache_embeddings


# =========================================================
# PROCURA A MENOR DISTÂNCIA ENTRE TODAS AS FACES
# =========================================================

def comparar_faces(
    faces_ref,
    faces_teste
):

    menor_distancia = float(
        "inf"
    )

    melhor_ref = None
    melhor_teste = None

    for face_ref in faces_ref:

        for face_teste in faces_teste:

            distancia = distancia_cosseno(
                face_ref["embedding"],
                face_teste["embedding"]
            )

            if distancia < menor_distancia:

                menor_distancia = distancia
                melhor_ref = face_ref
                melhor_teste = face_teste

    if (
        melhor_ref is None
        or melhor_teste is None
    ):

        raise ValueError(
            "Não foi possível comparar os embeddings."
        )

    return (
        menor_distancia,
        melhor_ref,
        melhor_teste
    )


# =========================================================
# ETAPA 3
# 512 COMPARAÇÕES RÁPIDAS
# =========================================================

def executar_comparacoes(
    cache_arcface,
    cache_facenet,
    barra=None,
    mensagem=None
):

    registros = []

    contador = 0

    caches = {
        "ArcFace": cache_arcface,
        "Facenet": cache_facenet
    }

    for modelo in MODELOS:

        cache_modelo = caches[
            modelo
        ]

        threshold = obter_threshold(
            modelo
        )

        for pessoa_ref, dados_ref in PESSOAS.items():

            for arquivo_ref in dados_ref[
                "refs"
            ]:

                caminho_ref = caminho_imagem(
                    pessoa_ref,
                    arquivo_ref
                )

                dados_ref_cache = cache_modelo.get(
                    caminho_ref,
                    {}
                )

                for pessoa_teste, dados_teste in PESSOAS.items():

                    mesma_pessoa = (
                        pessoa_ref
                        ==
                        pessoa_teste
                    )

                    tipo_comparacao = (
                        "Genuína"
                        if mesma_pessoa
                        else
                        "Impostora"
                    )

                    esperado = (
                        "Mesma pessoa"
                        if mesma_pessoa
                        else
                        "Pessoas diferentes"
                    )

                    for arquivo_teste in dados_teste[
                        "testes"
                    ]:

                        contador += 1

                        caminho_teste = caminho_imagem(
                            pessoa_teste,
                            arquivo_teste
                        )

                        dados_teste_cache = cache_modelo.get(
                            caminho_teste,
                            {}
                        )

                        if mensagem is not None:

                            mensagem.write(
                                f"Comparação "
                                f"{contador}/{TOTAL_EXECUCOES} — "
                                f"{nome_modelo(modelo)} | "
                                f"{arquivo_ref} × "
                                f"{arquivo_teste}"
                            )

                        status_ref = dados_ref_cache.get(
                            "status",
                            "Não processado"
                        )

                        status_teste = dados_teste_cache.get(
                            "status",
                            "Não processado"
                        )

                        faces_ref = dados_ref_cache.get(
                            "faces",
                            []
                        )

                        faces_teste = dados_teste_cache.get(
                            "faces",
                            []
                        )

                        if (
                            status_ref != "OK"
                            or
                            status_teste != "OK"
                            or
                            len(faces_ref) == 0
                            or
                            len(faces_teste) == 0
                        ):

                            erro = (
                                f"Referência: {status_ref}. "
                                f"Teste: {status_teste}."
                            )

                            registros.append({

                                "Modelo":
                                    nome_modelo(
                                        modelo
                                    ),

                                "Detector":
                                    DETECTOR,

                                "Metrica_distancia":
                                    DISTANCE_METRIC,

                                "Pessoa_referencia":
                                    pessoa_ref,

                                "Imagem_referencia":
                                    arquivo_ref,

                                "Faces_referencia":
                                    len(faces_ref),

                                "Cenario_referencia":
                                    cenario_faces(
                                        len(faces_ref)
                                    ),

                                "Pessoa_teste":
                                    pessoa_teste,

                                "Imagem_teste":
                                    arquivo_teste,

                                "Faces_teste":
                                    len(faces_teste),

                                "Cenario_teste":
                                    cenario_faces(
                                        len(faces_teste)
                                    ),

                                "Tipo_comparacao":
                                    tipo_comparacao,

                                "Esperado":
                                    esperado,

                                "Resultado":
                                    "Não avaliado",

                                "Distancia":
                                    None,

                                "Threshold":
                                    threshold,

                                "Classificacao":
                                    "N/A",

                                "Face_referencia_escolhida":
                                    None,

                                "Face_teste_escolhida":
                                    None,

                                "Face_ref_x":
                                    None,

                                "Face_ref_y":
                                    None,

                                "Face_ref_w":
                                    None,

                                "Face_ref_h":
                                    None,

                                "Face_teste_x":
                                    None,

                                "Face_teste_y":
                                    None,

                                "Face_teste_w":
                                    None,

                                "Face_teste_h":
                                    None,

                                "Status":
                                    "Não avaliado",

                                "Erro":
                                    erro
                            })

                        else:

                            try:

                                (
                                    distancia,
                                    melhor_ref,
                                    melhor_teste
                                ) = comparar_faces(
                                    faces_ref,
                                    faces_teste
                                )

                                verificado = (
                                    distancia
                                    <=
                                    threshold
                                )

                                classificacao = classificar(
                                    mesma_pessoa,
                                    verificado
                                )

                                area_ref = melhor_ref[
                                    "area"
                                ]

                                area_teste = melhor_teste[
                                    "area"
                                ]

                                registros.append({

                                    "Modelo":
                                        nome_modelo(
                                            modelo
                                        ),

                                    "Detector":
                                        DETECTOR,

                                    "Metrica_distancia":
                                        DISTANCE_METRIC,

                                    "Pessoa_referencia":
                                        pessoa_ref,

                                    "Imagem_referencia":
                                        arquivo_ref,

                                    "Faces_referencia":
                                        len(
                                            faces_ref
                                        ),

                                    "Cenario_referencia":
                                        cenario_faces(
                                            len(
                                                faces_ref
                                            )
                                        ),

                                    "Pessoa_teste":
                                        pessoa_teste,

                                    "Imagem_teste":
                                        arquivo_teste,

                                    "Faces_teste":
                                        len(
                                            faces_teste
                                        ),

                                    "Cenario_teste":
                                        cenario_faces(
                                            len(
                                                faces_teste
                                            )
                                        ),

                                    "Tipo_comparacao":
                                        tipo_comparacao,

                                    "Esperado":
                                        esperado,

                                    "Resultado":
                                        (
                                            "Mesma pessoa"
                                            if verificado
                                            else
                                            "Pessoas diferentes"
                                        ),

                                    "Distancia":
                                        distancia,

                                    "Threshold":
                                        threshold,

                                    "Classificacao":
                                        classificacao,

                                    "Face_referencia_escolhida":
                                        melhor_ref[
                                            "indice_face"
                                        ],

                                    "Face_teste_escolhida":
                                        melhor_teste[
                                            "indice_face"
                                        ],

                                    "Face_ref_x":
                                        area_ref.get(
                                            "x"
                                        ),

                                    "Face_ref_y":
                                        area_ref.get(
                                            "y"
                                        ),

                                    "Face_ref_w":
                                        area_ref.get(
                                            "w"
                                        ),

                                    "Face_ref_h":
                                        area_ref.get(
                                            "h"
                                        ),

                                    "Face_teste_x":
                                        area_teste.get(
                                            "x"
                                        ),

                                    "Face_teste_y":
                                        area_teste.get(
                                            "y"
                                        ),

                                    "Face_teste_w":
                                        area_teste.get(
                                            "w"
                                        ),

                                    "Face_teste_h":
                                        area_teste.get(
                                            "h"
                                        ),

                                    "Status":
                                        "Processado",

                                    "Erro":
                                        ""
                                })

                            except Exception as erro:

                                registros.append({

                                    "Modelo":
                                        nome_modelo(
                                            modelo
                                        ),

                                    "Detector":
                                        DETECTOR,

                                    "Metrica_distancia":
                                        DISTANCE_METRIC,

                                    "Pessoa_referencia":
                                        pessoa_ref,

                                    "Imagem_referencia":
                                        arquivo_ref,

                                    "Faces_referencia":
                                        len(
                                            faces_ref
                                        ),

                                    "Cenario_referencia":
                                        cenario_faces(
                                            len(
                                                faces_ref
                                            )
                                        ),

                                    "Pessoa_teste":
                                        pessoa_teste,

                                    "Imagem_teste":
                                        arquivo_teste,

                                    "Faces_teste":
                                        len(
                                            faces_teste
                                        ),

                                    "Cenario_teste":
                                        cenario_faces(
                                            len(
                                                faces_teste
                                            )
                                        ),

                                    "Tipo_comparacao":
                                        tipo_comparacao,

                                    "Esperado":
                                        esperado,

                                    "Resultado":
                                        "Não avaliado",

                                    "Distancia":
                                        None,

                                    "Threshold":
                                        threshold,

                                    "Classificacao":
                                        "N/A",

                                    "Face_referencia_escolhida":
                                        None,

                                    "Face_teste_escolhida":
                                        None,

                                    "Face_ref_x":
                                        None,

                                    "Face_ref_y":
                                        None,

                                    "Face_ref_w":
                                        None,

                                    "Face_ref_h":
                                        None,

                                    "Face_teste_x":
                                        None,

                                    "Face_teste_y":
                                        None,

                                    "Face_teste_w":
                                        None,

                                    "Face_teste_h":
                                        None,

                                    "Status":
                                        "Erro de comparação",

                                    "Erro":
                                        mensagem_erro(
                                            erro
                                        )
                                })

                        if barra is not None:

                            barra.progress(
                                contador
                                /
                                TOTAL_EXECUCOES
                            )

    df = pd.DataFrame(
        registros
    )

    df.to_csv(
        ARQUIVO_RESULTADOS,
        index=False,
        encoding="utf-8-sig"
    )

    return df


# =========================================================
# RESUMO DOS RESULTADOS
# =========================================================

def resumo_modelo(
    df,
    modelo
):

    dados = df[
        (df["Modelo"] == modelo)
        &
        (df["Status"] == "Processado")
    ]

    vp = (
        dados["Classificacao"] == "VP"
    ).sum()

    vn = (
        dados["Classificacao"] == "VN"
    ).sum()

    fp = (
        dados["Classificacao"] == "FP"
    ).sum()

    fn = (
        dados["Classificacao"] == "FN"
    ).sum()

    genuinas = vp + fn
    impostoras = fp + vn

    fmr = (
        fp / impostoras
        if impostoras > 0
        else 0
    )

    fnmr = (
        fn / genuinas
        if genuinas > 0
        else 0
    )

    total = (
        vp + vn + fp + fn
    )

    acuracia = (
        (vp + vn) / total
        if total > 0
        else 0
    )

    return {
        "VP": int(vp),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn),
        "FMR": float(fmr),
        "FNMR": float(fnmr),
        "Acuracia": float(acuracia)
    }


# =========================================================
# ABAS
# =========================================================

aba1, aba2, aba3 = st.tabs([
    "🧪 Experimento",
    "📊 Resultados",
    "ℹ️ Informações"
])


# =========================================================
# ABA EXPERIMENTO
# =========================================================

with aba1:

    st.header(
        "Experimento otimizado"
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Pessoas",
        4
    )

    c2.metric(
        "Imagens",
        40
    )

    c3.metric(
        "Modelos",
        2
    )

    c4.metric(
        "Comparações",
        512
    )

    st.success(
        "Nesta versão, o RetinaFace detecta cada imagem "
        "apenas uma vez. As 512 comparações utilizam "
        "embeddings previamente calculados."
    )

    st.info(
        "Imagens contendo várias pessoas são mantidas "
        "intencionalmente. Todas as faces detectadas são "
        "consideradas durante a comparação."
    )

    if os.path.exists(
        CACHE_DETECCAO
    ):

        st.write(
            "✅ Cache de detecção encontrado."
        )

    if os.path.exists(
        CACHE_ARCFACE
    ):

        st.write(
            "✅ Cache ArcFace encontrado."
        )

    if os.path.exists(
        CACHE_FACENET
    ):

        st.write(
            "✅ Cache FaceNet encontrado."
        )

    executar = st.button(
        "▶ Iniciar experimento otimizado",
        type="primary",
        use_container_width=True
    )

    if executar:

        # =============================================
        # ETAPA 1
        # =============================================

        st.subheader(
            "Etapa 1 — Detecção facial"
        )

        st.write(
            "RetinaFace analisando as 40 imagens. "
            "Resultados são salvos em cache."
        )

        barra_deteccao = st.progress(
            0
        )

        msg_deteccao = st.empty()

        cache_deteccao = detectar_faces_dataset(
            barra=barra_deteccao,
            mensagem=msg_deteccao
        )

        msg_deteccao.empty()

        df_faces = criar_csv_faces(
            cache_deteccao
        )

        st.success(
            "Etapa 1 concluída."
        )

        st.dataframe(
            df_faces,
            use_container_width=True,
            hide_index=True
        )

        # =============================================
        # ETAPA 2
        # =============================================

        st.subheader(
            "Etapa 2 — Embeddings ArcFace"
        )

        barra_arc = st.progress(
            0
        )

        msg_arc = st.empty()

        cache_arcface = gerar_embeddings_modelo(
            "ArcFace",
            cache_deteccao,
            barra=barra_arc,
            mensagem=msg_arc
        )

        msg_arc.empty()

        st.success(
            "Embeddings ArcFace concluídos."
        )

        # =============================================
        # ETAPA 3
        # =============================================

        st.subheader(
            "Etapa 3 — Embeddings FaceNet"
        )

        barra_face = st.progress(
            0
        )

        msg_face = st.empty()

        cache_facenet = gerar_embeddings_modelo(
            "Facenet",
            cache_deteccao,
            barra=barra_face,
            mensagem=msg_face
        )

        msg_face.empty()

        st.success(
            "Embeddings FaceNet concluídos."
        )

        # =============================================
        # ETAPA 4
        # =============================================

        st.subheader(
            "Etapa 4 — 512 comparações"
        )

        st.write(
            "Nesta etapa não é necessário executar "
            "RetinaFace novamente."
        )

        barra_comparacoes = st.progress(
            0
        )

        msg_comparacoes = st.empty()

        df_resultados = executar_comparacoes(
            cache_arcface,
            cache_facenet,
            barra=barra_comparacoes,
            mensagem=msg_comparacoes
        )

        msg_comparacoes.empty()

        st.success(
            "✅ Experimento concluído!"
        )

        st.balloons()

        st.write(
            f"Resultados salvos em "
            f"`{ARQUIVO_RESULTADOS}`"
        )


# =========================================================
# ABA RESULTADOS
# =========================================================

with aba2:

    st.header(
        "Resultados"
    )

    if not os.path.exists(
        ARQUIVO_RESULTADOS
    ):

        st.info(
            "O experimento final ainda não foi concluído."
        )

    else:

        df = pd.read_csv(
            ARQUIVO_RESULTADOS
        )

        processados = df[
            df["Status"]
            ==
            "Processado"
        ]

        falhas = df[
            df["Status"]
            !=
            "Processado"
        ]

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Execuções",
            len(df)
        )

        c2.metric(
            "Processadas",
            len(processados)
        )

        c3.metric(
            "Não avaliadas",
            len(falhas)
        )

        st.divider()

        for modelo in [
            "ArcFace",
            "FaceNet"
        ]:

            st.subheader(
                modelo
            )

            resumo = resumo_modelo(
                df,
                modelo
            )

            c1, c2, c3, c4 = st.columns(
                4
            )

            c1.metric(
                "VP",
                resumo["VP"]
            )

            c2.metric(
                "VN",
                resumo["VN"]
            )

            c3.metric(
                "FP",
                resumo["FP"]
            )

            c4.metric(
                "FN",
                resumo["FN"]
            )

            c5, c6, c7 = st.columns(
                3
            )

            c5.metric(
                "FMR",
                f"{resumo['FMR'] * 100:.2f}%"
            )

            c6.metric(
                "FNMR",
                f"{resumo['FNMR'] * 100:.2f}%"
            )

            c7.metric(
                "Acurácia",
                f"{resumo['Acuracia'] * 100:.2f}%"
            )

            st.divider()

        # =============================================
        # UMA FACE X MÚLTIPLAS FACES
        # =============================================

        st.header(
            "Uma face × múltiplas faces"
        )

        resumo_cenarios = []

        for modelo in [
            "ArcFace",
            "FaceNet"
        ]:

            dados_modelo = processados[
                processados[
                    "Modelo"
                ]
                ==
                modelo
            ]

            for cenario in [
                "Uma face",
                "Múltiplas faces"
            ]:

                dados = dados_modelo[
                    dados_modelo[
                        "Cenario_teste"
                    ]
                    ==
                    cenario
                ]

                vp = (
                    dados["Classificacao"]
                    ==
                    "VP"
                ).sum()

                vn = (
                    dados["Classificacao"]
                    ==
                    "VN"
                ).sum()

                fp = (
                    dados["Classificacao"]
                    ==
                    "FP"
                ).sum()

                fn = (
                    dados["Classificacao"]
                    ==
                    "FN"
                ).sum()

                genuinas = vp + fn
                impostoras = fp + vn

                fmr = (
                    fp / impostoras
                    if impostoras > 0
                    else 0
                )

                fnmr = (
                    fn / genuinas
                    if genuinas > 0
                    else 0
                )

                total = (
                    vp + vn + fp + fn
                )

                acc = (
                    (vp + vn) / total
                    if total > 0
                    else 0
                )

                resumo_cenarios.append({

                    "Modelo":
                        modelo,

                    "Cenário":
                        cenario,

                    "Comparações":
                        int(total),

                    "VP":
                        int(vp),

                    "VN":
                        int(vn),

                    "FP":
                        int(fp),

                    "FN":
                        int(fn),

                    "FMR (%)":
                        round(
                            fmr * 100,
                            2
                        ),

                    "FNMR (%)":
                        round(
                            fnmr * 100,
                            2
                        ),

                    "Acurácia (%)":
                        round(
                            acc * 100,
                            2
                        )
                })

        st.dataframe(
            pd.DataFrame(
                resumo_cenarios
            ),
            use_container_width=True,
            hide_index=True
        )

        # =============================================
        # MÚLTIPLAS FACES
        # =============================================

        st.header(
            "Imagens com múltiplas faces"
        )

        multiplas = processados[
            processados[
                "Faces_teste"
            ] > 1
        ]

        if len(
            multiplas
        ) == 0:

            st.info(
                "Nenhuma imagem de teste com "
                "múltiplas faces foi registrada."
            )

        else:

            colunas = [
                "Modelo",
                "Pessoa_referencia",
                "Imagem_referencia",
                "Pessoa_teste",
                "Imagem_teste",
                "Faces_teste",
                "Face_teste_escolhida",
                "Tipo_comparacao",
                "Resultado",
                "Distancia",
                "Threshold",
                "Classificacao"
            ]

            st.dataframe(
                multiplas[
                    colunas
                ],
                use_container_width=True
            )

        # =============================================
        # FALHAS
        # =============================================

        st.header(
            "Falhas / não avaliados"
        )

        if len(
            falhas
        ) == 0:

            st.success(
                "Nenhuma comparação ficou sem avaliação."
            )

        else:

            st.dataframe(
                falhas,
                use_container_width=True
            )

        # =============================================
        # TABELA COMPLETA
        # =============================================

        st.header(
            "Dados completos"
        )

        st.dataframe(
            df,
            use_container_width=True
        )


# =========================================================
# ABA INFORMAÇÕES
# =========================================================

with aba3:

    st.header(
        "Informações do experimento"
    )

    st.markdown(
        """
### Dataset

- **4 indivíduos**
- Barack Obama
- Neymar Jr.
- Scarlett Johansson
- Taylor Swift
- **2 imagens de referência por indivíduo**
- **8 imagens de teste por indivíduo**
- **40 imagens no total**

### Modelos

- ArcFace
- FaceNet

### Detecção

- RetinaFace
- Alinhamento facial ativado
- Detecção realizada uma única vez por imagem
- Todas as faces encontradas são preservadas

### Comparações

Por modelo:

- 64 comparações genuínas
- 192 comparações impostoras
- 256 comparações

Total:

- **512 comparações**

### Métrica

Distância cosseno (`cosine`)

### Classificações

- **VP** — Verdadeiro Positivo
- **VN** — Verdadeiro Negativo
- **FP** — Falso Positivo
- **FN** — Falso Negativo

### FMR

False Match Rate:

`FMR = FP / (FP + VN)`

### FNMR

False Non-Match Rate:

`FNMR = FN / (FN + VP)`

### Imagens com várias pessoas

Quando uma imagem contém mais de uma face,
todas as faces detectadas são mantidas.

Cada face da imagem de referência é comparada
com cada face da imagem de teste.

A menor distância facial encontrada é utilizada
para a decisão da comparação.

O índice e a área da face selecionada também
são registrados no CSV para permitir análise
posterior dos casos com múltiplas pessoas.
        """
    )
