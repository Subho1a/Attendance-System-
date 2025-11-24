# run placeholder
# run.py
from src.register import register_user
from src.train import train_svm
from src.recognize import recognize
import argparse

def main():
    parser = argparse.ArgumentParser(description="FaceNet + SVM Attendance System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register
    reg = subparsers.add_parser("register", help="Register a new user")
    reg.add_argument("--name", type=str, required=True, help="Name of the user")
    reg.add_argument("--n", type=int, default=50, help="Number of images to capture")

    # Train
    subparsers.add_parser("train", help="Train the SVM classifier")

    # Recognize
    rec = subparsers.add_parser("recognize", help="Run real-time attendance system")
    rec.add_argument("--threshold", type=float, default=0.75, help="Recognition confidence threshold")

    args = parser.parse_args()

    if args.command == "register":
        register_user(args.name, args.n)

    elif args.command == "train":
        train_svm()

    elif args.command == "recognize":
        recognize(threshold=args.threshold)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
