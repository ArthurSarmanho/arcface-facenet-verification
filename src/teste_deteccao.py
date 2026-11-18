from deepface import DeepFace

imagens = [
    "dataset/Barack_obama/teste5_obama.jpg",
    "dataset/Neymar_jr/teste8_neymar.jpg",
    "dataset/Taylor_swift/ref2_taylor.jpg",
    "dataset/Taylor_swift/teste2_taylor.jpg",
    "dataset/Taylor_swift/teste5_taylor.jpg",
]

for img in imagens:
    print("\n============================")
    print("IMAGEM:", img)

    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=True
        )

        print("OK - Face detectada")
        print("Quantidade de faces:", len(faces))

    except Exception as e:
        print("ERRO:", repr(e))
        print("TIPO:", type(e).__name__)
        print("CAUSA:", repr(e.__cause__))
        print("CONTEXTO:", repr(e.__context__))