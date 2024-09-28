This goal of this program is to detect any human that enters the frame and then matches their face with any known users stored in the database.

Any unknown faces will be highlighted red, whereas, known faces will be highlighted green.

To get this working for yourself: 

1. Run the code ```pip install -r requirements.txt``` this will install all necessary libraries.

2. Add a folder containing images of faces of people you want to be authorised. Store this in the root directory of the project. Ensure name of photos include the name of the person.

3. Update update the file path on line 20 in the initKnownFaces() func.

4. Update the block of code starting at line 34 in the initKnownFaces func. Right now they are set to user1 and user2.

5. Enjoy!

This is a personal project I've created just to practice logic taught to me from Uni, don't be surprised if it isn't the best working program.

[Created By: github.com/hijscripts]


