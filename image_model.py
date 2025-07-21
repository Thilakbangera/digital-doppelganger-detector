from deepface import DeepFace
import os

def compare_faces(file1_path: str, file2_path: str, model_name: str = "VGG-Face") -> dict:
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        return {"error": "One or both image paths are invalid."}
    
    try:
        result = DeepFace.verify(
            img1_path=file1_path,
            img2_path=file2_path,
            model_name=model_name,
            enforce_detection=False
        )
        
        return {
            "verified": result["verified"],
            "distance": result["distance"],
            "model": result["model"],
            "similarity_score": round((1 - result["distance"]) * 100, 2)
        }
    
    except Exception as e:
        return {"error": str(e)}
