import os
import psycopg2
from dotenv import load_dotenv
import numpy as np
import io
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create a database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def log_prediction(predicted_digit, confidence, true_label=None, image_data=None):
    """Log a prediction to the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Convert image data to binary if present
        binary_image = None
        if image_data is not None:
            # Convert numpy array to PNG bytes
            img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            binary_image = img_byte_arr.getvalue()
        
        # Insert prediction into database
        cur.execute(
            """
            INSERT INTO predictions 
            (predicted_digit, confidence, true_label, image_data)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (predicted_digit, confidence, true_label, binary_image)
        )
        
        inserted_id = cur.fetchone()[0]
        conn.commit()
        return inserted_id
        
    except Exception as e:
        print(f"Error logging prediction: {e}")
        conn.rollback()
        return False
        
    finally:
        cur.close()
        conn.close()

def update_true_label(prediction_id, true_label):
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            """
            UPDATE predictions
            SET true_label = %s
            WHERE id = %s
            """,
            (true_label, prediction_id)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating true label: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()

def get_prediction_stats():
    """Get statistics about predictions"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get total predictions
        cur.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cur.fetchone()[0]
        
        # Get accuracy (where true_label is not null)
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE predicted_digit = true_label) * 100.0 / 
                NULLIF(COUNT(*) FILTER (WHERE true_label IS NOT NULL), 0)
            FROM predictions 
            WHERE true_label IS NOT NULL
        """)
        accuracy = cur.fetchone()[0]
        
        # Get distribution of predictions
        cur.execute("""
            SELECT predicted_digit, COUNT(*)
            FROM predictions
            GROUP BY predicted_digit
            ORDER BY predicted_digit
        """)
        distribution = cur.fetchall()
        
        # Debug query for correctness verification
        cur.execute("""
            SELECT predicted_digit, true_label, COUNT(*)
            FROM predictions
            WHERE true_label IS NOT NULL
            GROUP BY predicted_digit, true_label
            ORDER BY predicted_digit, true_label
        """)
        detailed_stats = cur.fetchall()
        print("Detailed prediction stats:")
        for stat in detailed_stats:
            print(f"Predicted: {stat[0]}, True: {stat[1]}, Count: {stat[2]}")
            
        result = {
            'total_predictions': total_predictions,
            'accuracy': accuracy if accuracy is not None else 0.0,
            'distribution': dict(distribution)
        }
        print(f"Stats result: {result}")
        return result
        
    finally:
        cur.close()
        conn.close()

def get_prediction_history():
    """Retrieve the most recent predictions."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, predicted_digit, confidence, true_label, timestamp
            FROM predictions
            ORDER BY timestamp DESC
        """)
        history = cur.fetchall()
        return history
    finally:
        cur.close()
        conn.close()