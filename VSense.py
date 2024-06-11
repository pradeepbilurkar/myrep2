import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import rembg


class OpenCVGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Set initial window size
        self.window.geometry("800x600")

        # Create a frame for the video player controls at the top of the canvas
        self.video_controls_frame = tk.Frame(self.window)
        self.video_controls_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Create canvas to display images
        self.canvas = tk.Canvas(self.window, width=1200, height=400)
        self.canvas.pack()

        # Create a frame for the buttons at the bottom of the canvas
        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Buttons
        self.btn_open = tk.Button(self.button_frame, text="Open Image", command=self.open_image)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_gray = tk.Button(self.button_frame, text="Convert to Grayscale", command=self.convert_to_grayscale)
        self.btn_gray.pack(side=tk.LEFT)

        self.btn_detect = tk.Button(self.button_frame, text="Detect Blobs and Contours",
                                    command=self.detect_blobs_and_contours)
        self.btn_detect.pack(side=tk.LEFT)

        # Button to toggle drawing mode for circle
        self.drawing_mode_circle = False
        self.btn_draw_circle = tk.Button(self.button_frame, text="Draw Circle", command=self.toggle_drawing_mode_circle)
        self.btn_draw_circle.pack(side=tk.LEFT)

        # Button to toggle drawing mode for rectangle
        self.drawing_mode_rectangle = False
        self.btn_draw_rectangle = tk.Button(self.button_frame, text="Draw Rectangle",
                                            command=self.toggle_drawing_mode_rectangle)
        self.btn_draw_rectangle.pack(side=tk.LEFT)

        self.btn_draw_line = tk.Button(self.button_frame, text="Draw Line", command=self.draw_line)
        self.btn_draw_line.pack(side=tk.LEFT)

        self.btn_threshold = tk.Button(self.button_frame, text="Threshold", command=self.threshold_image)
        self.btn_threshold.pack(side=tk.LEFT)

        self.btn_reset = tk.Button(self.button_frame, text="Reset", command=self.reset)
        self.btn_reset.pack(side=tk.LEFT)

        # Add a button to remove background
        btn_remove_bg = tk.Button(self.button_frame, text="Remove Background", command=self.remove_background)
        btn_remove_bg.pack(pady=5)

        # Add a button to add background

        # Add buttons for OpenCV operations
        btn_add_bg = tk.Button(self.button_frame, text="Add Background", command=self.open_background_image)
        btn_add_bg.pack(side=tk.LEFT)

        btn_add_fg = tk.Button(self.button_frame, text="Add Foreground", command=self.open_foreground_image)
        btn_add_fg.pack(side=tk.LEFT)

        self.image = None
        self.blob_image = None
        self.contour_image = None

        # Initialize image and bind mouse events
        self.image = None
        self.start_point = None
        self.circle = None

        # circle
        self.canvas.bind("<Button-1>", self.draw_circle_with_mouse)
        self.canvas.bind("<B1-Motion>", self.draw_circle_with_mouse)
        self.canvas.bind("<ButtonRelease-1>", self.draw_circle_with_mouse)

        # Initialize rectangle
        self.rectangle = None

        # rectangle
        # Bind mouse events for drawing rectangles
        self.canvas.bind("<Button-3>", self.on_canvas_click)
        self.canvas.bind("<B3-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_canvas_release)

        # Button to select video file
        self.btn_select_video = tk.Button(self.video_controls_frame, text="Select Video", command=self.select_video)
        self.btn_select_video.pack(side=tk.LEFT)

        # Button to select background image
        self.btn_select_bg = tk.Button(self.video_controls_frame, text="Select Background Image",
                                       command=self.select_background)
        self.btn_select_bg.pack(side=tk.LEFT, padx=5)

        # Button to start/stop video
        self.btn_start_stop = tk.Button(self.video_controls_frame, text="Start", command=self.toggle_video)
        self.btn_start_stop.pack(side=tk.LEFT, padx=5)

        # Create canvas to display video
        self.video_canvas = tk.Canvas(self.window, width=800, height=400)  # Adjusted canvas size
        self.video_canvas.pack()

        self.vid = None
        self.bg_img = None
        self.fps = None
        self.seg = SelfiSegmentation()
        self.out = None
        self.is_playing = False
        self.update_id = None

        self.window.mainloop()

        self.original_photo = None
        self.gray_photo = None

        self.image = None
        self.blob_image = None
        self.contour_image = None

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Resize the image to fit the canvas
        original_image = Image.fromarray(image_rgb).resize((400, 400))  # Adjusted image size
        # Convert the image to a Tkinter-compatible format
        self.original_photo = ImageTk.PhotoImage(image=original_image)
        # Display the original image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo)

    def convert_to_grayscale(self):
        if self.image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Resize the images to fit within the canvas
            original_pil = self.resize_image(Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)), width=380,
                                             height=300)
            gray_pil = self.resize_image(Image.fromarray(gray_image), width=380, height=300)

            # Convert images to Tkinter-compatible format
            original_tk = ImageTk.PhotoImage(original_pil)
            gray_tk = ImageTk.PhotoImage(gray_pil)

            # Display both images on the canvas with space between them
            self.canvas.delete("all")
            self.canvas.create_image(10, 10, anchor='nw', image=original_tk)
            self.canvas.create_image(original_tk.width() + 20, 10, anchor='nw', image=gray_tk)

            # Keep references to avoid garbage collection
            self.original_tk = original_tk
            self.gray_tk = gray_tk

    def resize_image(self, image, width, height):
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_width = width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = height
            new_width = int(new_height * aspect_ratio)
        return image.resize((new_width, new_height))

    def display_images(self):
        if self.image is not None and self.blob_image is not None and self.contour_image is not None:
            # Convert images to RGB format
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            blob_image_rgb = cv2.cvtColor(self.blob_image, cv2.COLOR_BGR2RGB)
            contour_image_rgb = cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB)

            # Resize images to fit canvas
            image_pil = Image.fromarray(image_rgb).resize((400, 400))
            blob_image_pil = Image.fromarray(blob_image_rgb).resize((400, 400))
            contour_image_pil = Image.fromarray(contour_image_rgb).resize((400, 400))

            # Merge images horizontally
            merged_image = Image.new("RGB", (1200, 400))
            merged_image.paste(image_pil, (0, 0))
            merged_image.paste(blob_image_pil, (400, 0))
            merged_image.paste(contour_image_pil, (800, 0))

            # Display merged image on canvas
            self.photo = ImageTk.PhotoImage(image=merged_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def detect_blobs_and_contours(self):
        if self.image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Detect blobs in the grayscale image
            params = cv2.SimpleBlobDetector_Params()
            params.filterByCircularity = False
            params.filterByInertia = False
            params.filterByConvexity = False
            params.filterByArea = True
            params.minArea = 100
            params.maxArea = 2000
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray_image)

            # Draw blobs on the image
            self.blob_image = cv2.drawKeypoints(self.image.copy(), keypoints, None, (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Detect contours around the blobs
            blob_mask = np.zeros_like(gray_image)
            for keypoint in keypoints:
                x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                cv2.circle(blob_mask, (x, y), int(keypoint.size / 2), (255), -1)
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the image with blobs
            self.contour_image = self.image.copy()
            cv2.drawContours(self.contour_image, contours, -1, (0, 255, 0), 2)

            # Display images
            self.display_images()

    def reset(self):
        self.image = None
        self.canvas.delete("all")
        # Reset the drawing mode and update the button text accordingly
        self.drawing_mode_circle = False
        self.btn_draw_circle.config(text="Draw Circle")
        self.drawing_mode_rectangle = False
        self.btn_draw_rectangle.config(text="Draw Rectangle")

    def toggle_drawing_mode_circle(self):
        self.drawing_mode_circle = not self.drawing_mode_circle
        if self.drawing_mode_circle:
            self.drawing_mode_rectangle = False  # Deactivate rectangle drawing mode
            self.btn_draw_circle.config(text="Drawing Mode ON")
            self.btn_draw_rectangle.config(text="Draw Rectangle")
        else:
            self.btn_draw_circle.config(text="Draw Circle")

    def toggle_drawing_mode_rectangle(self):
        self.drawing_mode_rectangle = not self.drawing_mode_rectangle
        if self.drawing_mode_rectangle:
            self.drawing_mode_circle = False  # Deactivate circle drawing mode
            self.btn_draw_rectangle.config(text="Drawing Mode ON")
            self.btn_draw_circle.config(text="Draw Circle")
        else:
            self.btn_draw_rectangle.config(text="Draw Rectangle")

    def draw_circle_with_mouse(self, event):
        if self.drawing_mode_circle:
            if event.type == tk.EventType.ButtonPress:
                self.start_point = (event.x, event.y)
                # Create a new circle on the canvas at the start point
                self.circle = self.canvas.create_oval(event.x, event.y, event.x, event.y, outline='green')
            elif event.type == tk.EventType.Motion:
                # Update the circle to match the current mouse position while dragging
                self.canvas.coords(self.circle, self.start_point[0], self.start_point[1], event.x, event.y)
            elif event.type == tk.EventType.ButtonRelease:
                # No need to do anything on release
                pass

    def draw_rectangle_with_mouse(self, event):
        if self.drawing_mode_rectangle:
            if event.type == tk.EventType.ButtonPress:
                self.start_point = (event.x, event.y)
                # Create a new rectangle on the canvas at the start point
                self.rectangle = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='blue')
            elif event.type == tk.EventType.Motion:
                # Update the rectangle to match the current mouse position while dragging
                self.canvas.coords(self.rectangle, self.start_point[0], self.start_point[1], event.x, event.y)
            elif event.type == tk.EventType.ButtonRelease:
                # No need to do anything on release
                pass

    def draw_line(self):
        if self.image is not None:
            x1 = simpledialog.askinteger("Draw Line", "Enter start X coordinate:")
            y1 = simpledialog.askinteger("Draw Line", "Enter start Y coordinate:")
            x2 = simpledialog.askinteger("Draw Line", "Enter end X coordinate:")
            y2 = simpledialog.askinteger("Draw Line", "Enter end Y coordinate:")
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                color = (0, 255, 0)  # Green color
                thickness = 2  # Thickness of the line
                cv2.line(self.image, (x1, y1), (x2, y2), color, thickness)
                self.display_image()

    def on_canvas_click(self, event):
        if self.drawing_mode_rectangle:
            self.start_point = (event.x, event.y)
            # Create a rectangle at the start point
            self.rectangle = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='blue')

    def on_canvas_drag(self, event):
        if self.drawing_mode_rectangle and self.start_point:
            # Update rectangle dimensions as the mouse is dragged
            self.canvas.coords(self.rectangle, self.start_point[0], self.start_point[1], event.x, event.y)

    def on_canvas_release(self, event):
        if self.drawing_mode_rectangle:
            # Reset start point when mouse is released
            self.start_point = None

    def threshold_image(self):
        if self.image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Apply thresholding
            _, thresholded_image = cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY)
            # Display the thresholded image
            self.image = thresholded_image
            self.display_image()

    def remove_background(self):
        # Get the input image path from the user
        input_image_path = filedialog.askopenfilename(title="Select Input Image",
                                                      filetypes=[("Image Files", "*.jpg; *.jpeg; *.png")])
        if not input_image_path:
            return

        try:
            # Load the input image
            input_image = Image.open(input_image_path)

            # Remove the background using U-2-Net
            output_image = rembg.remove(input_image)

            # Save the output image
            output_image_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                             filetypes=[("PNG Files", "*.png")])
            if output_image_path:
                output_image.save(output_image_path)
                messagebox.showinfo("Success",
                                    "Background removed successfully and saved as {}".format(output_image_path))
        except Exception as e:
            messagebox.showerror("Error", "An error occurred: {}".format(str(e)))

    def open_background_image(self):
        self.background_image_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image Files", "*.jpg; *.jpeg; *.png")]
        )
        if self.background_image_path:
            print("Background Image Path:", self.background_image_path)
            self.process_image(self.background_image_path, is_background=True)

    def open_foreground_image(self):
        self.foreground_image_path = filedialog.askopenfilename(
            title="Select Foreground Image",
            filetypes=[("Image Files", "*.jpg; *.jpeg; *.png")]
        )
        if self.foreground_image_path:
            print("Foreground Image Path:", self.foreground_image_path)
            self.process_image(self.foreground_image_path, is_background=False)

    def process_image(self, image_path, is_background):
        try:
            # Load the image
            image = Image.open(image_path)

            # Resize the image to fit the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            image = image.resize((canvas_width, canvas_height))

            # Convert the image to a format compatible with Tkinter
            photo = ImageTk.PhotoImage(image=image)

            if is_background:
                # Clear canvas before displaying new background image
                self.canvas.delete("background")
                self.background_photo = photo
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_photo, tags="background")
            else:
                # Clear canvas before displaying new foreground image
                self.canvas.delete("foreground")
                self.foreground_photo = photo
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.foreground_photo, tags="foreground")

        except Exception as e:
            messagebox.showerror("Error", "An error occurred: {}".format(str(e)))

    def show_image(self, image):
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # keep a reference!

    def reset(self):
        self.image = None
        self.canvas.delete("all")

        # Add methods for video player controls (select_video, select_background, toggle_video, update_gui, restart_video)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if file_path:
            if self.vid is not None:
                self.vid.release()  # Release the previous video capture
            self.vid = cv2.VideoCapture(file_path)
            self.vid.set(3, 1600)
            self.vid.set(4, 800)
            self.fps = self.vid.get(cv2.CAP_PROP_FPS)
            self.out = None  # Reset the video writer
            print("Video loaded successfully:", file_path)
            if self.is_playing:
                self.toggle_video()  # Stop video if currently playing
                self.toggle_video()  # Start video again to update with new file

    def select_background(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if file_path:
            self.bg_img = cv2.imread(file_path)
            self.bg_img = cv2.resize(self.bg_img, (640, 480))  # Resize background image
            self.out = None  # Reset the video writer
            print("Background image loaded successfully:", file_path)

    def update_gui(self):
        ret, video = self.vid.read()

        if ret:
            print("Frame read successfully")
            video_resized = cv2.resize(video, (640, 480))
            vid_rmbg = self.seg.removeBG(video_resized, self.bg_img)
            vid_rmbg = cv2.cvtColor(vid_rmbg, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(vid_rmbg)
            img_tk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.img = img_tk  # Save a reference to avoid garbage collection

            # Initialize self.out if it is not already initialized
            if self.out is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

            self.out.write(vid_rmbg)
        else:
            # Video ended, reset the video capture
            self.vid.release()
            self.vid = None
            self.is_playing = False
            self.btn_start_stop.config(text="Start")
            self.window.after_cancel(self.update_id)
            print("Video playback ended")

        if self.is_playing:
            self.update_id = self.window.after(int(1000 / self.fps), self.update_gui)

    def toggle_video(self):
        if self.vid is None:
            messagebox.showerror("Error", "No video selected!")
            return

        if self.bg_img is None:
            messagebox.showerror("Error", "No background image selected!")
            return

        if not self.is_playing:
            # Start playing video
            self.is_playing = True
            self.btn_start_stop.config(text="Stop")
            self.update_gui()  # Start video playback
        else:
            # Stop playing video
            self.is_playing = False
            self.btn_start_stop.config(text="Start")
            if self.update_id is not None:
                self.window.after_cancel(self.update_id)  # Cancel the update job

    def restart_video(self, event):
        if self.vid is None:
            return
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.is_playing = True
        self.btn_start_stop.config(text="Stop")
        self.update_gui()  # Start video playback


# Create a window and pass it to the Application object
root = tk.Tk()
app = OpenCVGUI(root, "OpenCV GUI")
root.mainloop()
