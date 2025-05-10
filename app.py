import cv2
from ultralytics import solutions

# Load the video
video = cv2.VideoCapture("demo.mp4")

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = 1020  # Resized width
height = 500  # Resized height

# Initialize video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# Define the region for counting (a narrow rectangle where packets cross)
counting_region = [(600, 280), (600, 320), (300, 320), (300, 280)]

# Initialize the object counter
packet_counter = solutions.ObjectCounter(
    region=counting_region,
    model="best.pt",
    show_in=True,
    show_out=False,
    line_width=2,
)

# Create a window to display the video
cv2.namedWindow("Packet Counter")

# Variable to keep track of the total number of packets counted
total_packets = 0

# Process the video frame by frame
while True:
    success, frame = video.read()
    if not success:
        break

    # Resize the frame for display
    frame = cv2.resize(frame, (width, height))

    # Count packets in the frame
    frame = packet_counter.count(frame)

    # Get the current count
    current_count = packet_counter.in_count

    # Update total packets if new packets are detected
    if current_count > total_packets:
        total_packets = current_count

    # Save the frame to the output video
    out.write(frame)

    # Show the frame with the count
    cv2.imshow("Packet Counter", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()

# Print the final count
print(f"Final Packet Count: {total_packets}")