from app import app
from extensions import db

if __name__ == "__main__":
    try:
        with app.app_context():
            db.create_all()
        print("✅ Tablas creadas/actualizadas en la base de datos indicada por DATABASE_URL")
    except Exception as e:
        print(f"Error al crear tablas: {e}")