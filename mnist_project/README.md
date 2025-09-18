# MNIST Digit Recognizer

A full-stack web application that allows users to draw digits which are then classified using a PyTorch deep learning model. The application is containerized with Docker and includes a PostgreSQL database for logging predictions.

## [Live Demo](http://188.245.253.11:8501/)

## Project Overview

This project is a complete end-to-end machine learning application that:

1. Uses a PyTorch CNN model trained on the MNIST dataset
2. Provides a web interface built with Streamlit where users can:
   - Draw digits on a canvas
   - Get real-time predictions
   - Provide feedback on prediction accuracy
3. Logs all predictions to a PostgreSQL database
4. Is fully containerized with Docker for easy deployment

## Features

- **Interactive Drawing Canvas**: Draw digits using a simple and intuitive interface
- **Real-time Predictions**: Get immediate feedback on what digit the model thinks you've drawn
- **Confidence Metrics**: See how confident the model is about its prediction
- **User Feedback**: Contribute to model improvement by providing the true label
- **Prediction History**: View past predictions and their accuracy
- **Performance Statistics**: See overall model performance metrics

## Technical Stack

- **Machine Learning**: PyTorch
- **Frontend**: Streamlit, Streamlit-drawable-canvas
- **Database**: PostgreSQL
- **Containerization**: Docker, Docker Compose
- **Deployment**: Self-managed server (Hetzner)

## Project Structure

```
.
├── src
│   ├── db
│   │   └── database.py
│   ├── model
│   │   ├── model.py
│   │   ├── train.py
│   │   └── saved
│   │       └── mnist_model.pth
│   └── web
│       └── app.py
├── .env
├── docker-compose.yml
├── Dockerfile
├── init.sql
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Running Locally

1. Clone the repository:
   ```
   git clone https://github.com/your-username/mnist-digit-recognizer.git
   cd mnist-digit-recognizer
   ```

2. Create a `.env` file with your database credentials (use `.env.example` as a template)

3. Build and start the containers:
   ```
   docker-compose up -d
   ```

4. Access the web interface at `http://localhost:8501`

### Training the Model (Optional)

If you want to retrain the model:

```
docker exec -it mnist_web python src/model/train.py
```

## Deployment

The application is deployed on a self-managed Hetzner server. To deploy on your own server:

1. Set up a server with Docker and Docker Compose installed
2. Clone the repository to your server
3. Follow the same steps as for local deployment
4. Configure any necessary firewall rules to allow traffic to port 8501

## Database Schema

The PostgreSQL database contains a single `predictions` table with the following schema:

| Column          | Type      | Description                       |
|-----------------|-----------|-----------------------------------|
| id              | SERIAL    | Primary key                       |
| predicted_digit | INTEGER   | The digit predicted by the model  |
| confidence      | FLOAT     | Prediction confidence (0-100%)    |
| true_label      | INTEGER   | User-provided correct label       |
| image_data      | BYTEA     | Binary image data                 |
| timestamp       | TIMESTAMP | When the prediction was made      |

## Model Architecture

The MNIST classifier uses a convolutional neural network (CNN) with:

- Two convolutional layers (32 and 64 channels)
- Max pooling
- Dropout for regularization
- Two fully connected layers
- 10-class output with softmax activation

The model achieves approximately 99% accuracy on the MNIST test set.

## Future Improvements

- [ ] Add user authentication
- [ ] Implement model retraining based on user feedback
- [ ] Create an API endpoint for programmatic access
- [ ] Add more visualization options for model performance
- [ ] Implement responsive design for mobile users

## License

[MIT License](LICENSE)

## Acknowledgments

- The MNIST dataset providers
- PyTorch team
- Streamlit team