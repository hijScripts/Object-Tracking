This is a personal project I started to detect any human that enters the frame and then matches their face with any known users stored in the database.

Any unknown faces will be highlighted red, whereas, known faces will be highlighted green.

To get this working for yourself: 

1. Run the code ```pip install -r requirements.txt``` this will install all necessary libraries.

2. Add a folder containing images of faces of people you want to be authorised. Store this in the root directory of the project. Ensure name of photos include the name of the person.

3. Update update the file path on line 20 in the initKnownFaces() func.

4. Update the block of code starting at line 34 in the initKnownFaces func. Right now they are set to user1 and user2.

5. Run the main file and enjoy!

Any questions, please feel free to contact me on any platform.

I've also included a generateImages file which has a block of code allowing you to take photos using your webcam. "Esc" closes the program and "s" takes a photo and stores it in a folder it creates.

[Created By: github.com/hijscripts]


