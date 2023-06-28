<a name="readme-top"></a>

<!-- START OF DOCUMENT -->

# lion-city-soundscapes-visualisation

This repository stores the code to build a Django web application for the interactive <a href="https://leafletjs.com">Leaflet</a> map described in the following publication:

> Kenneth Ooi, Jessie Goh, Hao-Weng Lin, Zhen-Ting Ong, Trevor Wong, Karn Watcharasupat, Bhan Lam, Woon-Seng Gan, "Lion City Soundscapes (LCS): Recording and Validation of Characteristic Soundscapes of Singapore", in _29th International Congress on Sound and Vibration, ICSV 2023_, Prague, Czech Republic, 2023.

A (temporarily working) deployed example of the interactive map can be accessed at http://18.136.105.85/map/.

# Related resources

- The publication will be accessible at https://hdl.handle.net/10356/168104 after ICSV 2023 ends on July 13, 2023.
- For the version of the project repository submitted for Jessie's Final Year Project, please refer to https://github.com/nemopotatoes/Lion-City-Soundscapes-Visualisation.
- The Lion City Soundscapes (LCS) dataset itself is accessible at https://doi.org/10.21979/N9/AVHSBX.
- The replication code for the statistical analysis and excerption methodology for the LCS dataset is accessible at https://github.com/ntudsp/lion-city-soundscapes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Getting started

## Step 1: Downloading this repository

Firstly, clone this repository by manually downloading it from https://github.com/ntudsp/lion-city-soundscapes-visualisation, or enter the following line from a terminal (you need to have <a href="https://github.com/git-guides/install-git">git</a> installed first, of course):

  ```sh
  git clone https://github.com/ntudsp/lion-city-soundscapes-visualisation
  ```

You may then navigate to the downloaded folder with 

  ```sh
  cd lion-city-soundscapes-visualisation
  ```
	
## Step 2: Setting up the project environment

Next, you will need to set up the project environment to run the application. We provide three alternatives in this section, and you may choose whichever is the most convenient (but not more than one at a time, of course).

### Alternative A: With conda installation commands
<a name="alt-A"></a>

If you are using <a href="https://www.anaconda.com/download">conda</a>, you may run the following commands in sequence to create the environment and install the necessary packages (note that the `-n` option specifies the name of the environment to be created):

  ```sh
  conda create -n lcs-vis
  conda activate lcs-vis
  conda install django
  conda install python-dotenv
  ```
	
For reference, we did this on conda version 23.3.1 (as of 28 June 2023). The above commands installed Django version 4.1 and python-dotenv version 0.21.0.

### Alternative B: With conda environment file
<a name="alt-B"></a> 

If you are using <a href="https://www.anaconda.com/download">conda</a>, you may also run the following commands to create and activate the environment from the `.yml` file in this repository (note that the `-f` option specifies the file containing the environment specifications and the `-n` option specifies the name of the environment to be created):

  ```sh
  conda env create -f lcs-vis.yml -n lcs-vis
  conda activate lcs-vis
  ```

### Alternative C: With `venv`
<a name="alt-C"></a>

First, ensure that <a href="https://www.python.org/downloads/">Python</a> and pip (a Python package manager) are installed on your system. Our system had Python version 3.10.9 and pip version 22.3.1 installed. You may check the versions installed on your system with the following commands:

  ```sh
  python --version
  pip --version
  ```
   
Then, create a virtual environment named `lcs-vis` using `venv` (note that the `-m` option runs library modules as scripts):

  ```sh
  python -m venv lcs-vis
  ```
   
Then, go to the virtual environment's directory and activate it:

  ```sh
  cd lcs-vis\Scripts
  activate
  ```
 
Then, go back to the root directory and install all other requirements for the application (note that the `-r` option tells pip to install from the specified requirements file):

  ```sh
  cd ..\..
  pip install -r requirements.txt
  ```
   
## Step 3: Running the web application

Once the project environment has been set up correctly, the following command will start the (local) server hosting the web application:

  ```sh
  python manage.py runserver
  ```
	
Alternatively, in case your system doesn't load the static files (e.g., CSS, CSV files) for the application automatically, you may want to run this instead (this will create a new folder called `static` in the root folder of this repository that django may try to look for, depending on your system settings):

  ```sh
  python manage.py collectstatic
  python manage.py runserver
  ```
	
The application will then be accessible at http://127.0.0.1:8000/ (you may enter this address into any web browser, such as Firefox or Chrome, to view it). To stop the server, you may press `Ctrl`+`C` on your keyboard in the same terminal that was used to start it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Directory structure

## This repository (after cloning with `git` and/or after running <a href="#alt-A">Alternative A</a> or <a href="#alt-B">Alternative B</a>)

    .
    ├── lioncitysoundscapes # Application folder containing HTML files, webpage format, and current app URL. Note that django projects may each contain multiple apps, each with their own folder, but our project only has one app (lioncitysoundscapes), thus there is only one folder here.
	│   ├── ...             
    │   ├── urls.py         # Defines paths to webpages themselves.
	│   └── views.py        # Defines functions to make HTTP requests to handle page loading.
	│                                
	├── staticfiles         # The folder that we configured ("forced") django to try to look for to load the application (cf. Lines 132-134 in .\visualisation\settings.py). However, depending on your own system settings, django may try to look for a folder named `static` instead (cf. Line 129 in .\visualisation\settings.py), in which case the `python manage.py collectstatic` (as described in Step 3) needs to be run before starting the server.
	│   ├── css
    │   ├── csv
    │   ├── images     
	│   └── js
	│
    ├── visualisation       # Configuration files for the application                        
	├── ...
	├── README.md           # This file.
	├── lcs-vis.yml         # The Anaconda environment containing required packages to build the application using Alternative B.
	├── manage.py           # Python file to run and manage the (local) application server
    └── requirements.txt    # The requirements file for setting up the project environment using Alternative C
	
## This repository (after running <a href="#alt-C">Alternative C</a>)

    .
    ├── lcs-vis             # `venv` environment folder
	│   ├── Include 
    │   ├── Lib  
    │   ├── Scripts
	│   │   ├── activate    # Run in command line/terminal to activate virtual environment
	│   │   └── ...
	│   │	
	│   └── pyenv.cfg
	│
    ├── lioncitysoundscapes
	│   ├── ...             
    │   ├── urls.py                   
	│   └── views.py  
	│                                
	├── staticfiles
	│   ├── css
    │   ├── csv
    │   ├── images     
	│   └── js
	│
    ├── visualisation                               
	├── ...
	├── README.md
	├── lcs-vis.yml
	├── manage.py
    └── requirements.txt

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Basic customisation

Here, we outline some basic methods to customise the map layout based on the code in this repository, if you would like to repurpose it for your own context. If the project server (i.e., the one at http://127.0.0.1:8000/) is running on your browser while you are making the changes, you may need to force a reload of the static files and cache on your browser to make the application reflect those changes in your browser (on Firefox and Chrome, this can be performed with the shortcut `Ctrl`+`Shift`+`R`).

## Map layer

For this project, we use the <a href="https://www.onemap.gov.sg/">OneMap API</a>, which generates a detailed map of Singapore, but is limited to Singapore only. If you would like to show other regions of the world, you may want to consider using the <a href="https://www.openstreetmap.org">OpenStreetMap API</a>, which covers the entire world.

This may be done by changing the tile provider in `.\staticfiles\js\map.js` to the one for OpenStreetMap, and the corresponding attribution statement. In addition, we also recommend a change in the min/max zoom levels, maximum bounds of the map, and setting of the `noWrap` property to `true` to maintain a relatively "nice" view of the larger world map. Specifically, this may done by modifying Lines 121-125 in `.\staticfiles\js\map.js` from

  ```JavaScript
  121 var basemap = L.tileLayer('https://maps-{s}.onemap.sg/v3/Grey/{z}/{x}/{y}.png', {
  122     detectRetina: true,
  123     maxZoom: 20,
  124     minZoom: 11,
  125     attribution: '<a href="https://www.onemap.gov.sg/home/">OneMap</a> | Map data &copy; contributors, <a href="http://SLA.gov.sg">Singapore Land Authority</a>',
  ...
  ...
  135 map.setMaxBounds([[1.50073, 104.1147], [1.16, 103.602]]);
  ```
  
to

  ```JavaScript
  121 var basemap = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  122     detectRetina: true,
  123     maxZoom: 20,
  124     minZoom: 2,
  125     attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  126     noWrap: true,
  ...
  ...
  136 map.setMaxBounds([[-90, -180], [90, 180]]);
  ```

Note that `minZoom` controls the largest possible area of the world that can be seen by zooming out (the smaller the number, the larger the area that can be seen at once). The `setMaxBounds` method defines the top right and bottom left coordinates of the rectangle defining the maximum boundary of the map, i.e., beyond which the map will snap back to the original position.

## Initial position & zoom level

Currently, the map is configured to initially show the entirety of Singapore. However, you may wish to change this by modifying the starting latitude (1.350270), longitude (103.829959), and zoom level (12) of `.\staticfiles\js\map.js` at Line 4:

  ```JavaScript
  002 var map = L.map('mapdiv', {
  003     zoomControl: false                  // remove top left zoom controls
  004 }).setView([1.350270, 103.829959],12);  // lat and long coordinates + initial zoom
  ```

## Adding markers

The markers on the map are dynamically added to the map based on the information provided in the CSV file at `.\staticfiles\csv\locations.csv`. Each row in the CSV file will correspond to one marker position on the interactive map, so you may add or remove rows as necessary if you wish to show a different set of markers for your use case. The column headers with their meanings in the context of our visualisation are as follows:

- `Title` : strings
  - A brief description of the actual location where the soundscape was recorded for the LCS dataset, which we refer to as the "recording location".
  - This may differ from the exact location identified by the <a href="https://doi.org/10.3390/su14127485">Singapore Soundscape Site Selection Survey</a> (S5 study), which we refer to as the "S5 location". This is due to the fact that the S5 location may be physically or publicly inaccessible (e.g., on water).
- `Latitude` : floating point numbers
  - The latitude of the recording location (i.e., the actual location where the soundscape was recorded for the LCS dataset).
- `Longitude` : floating point numbers
  - The longitude of the recording location (i.e., the actual location where the soundscape was recorded for the LCS dataset).
- `Type` : strings in {"F&E", "C&R", "C&T", "B&L"}
  - The perceptual quadrant in which the soundscape at the S5 location was classified using the modified $k$-means clustering algorithm in the <a href="https://doi.org/10.3390/su14127485">S5 study</a>.
  - In the interactive map, these also control the colour of the marker assigned to the point on the map.
  - Keys:
    - `F&E` : Full of life and exciting (orange)
	- `C&R` : Chaotic and restless (red)
	- `C&T` : Calm and tranquil (green)
	- `B&L` : Boring and lifeless (black)
- `vidEmbed` : strings
  - The YouTube embed link to the 1-minute excerpt of the full-length recording from the Insta360 One R camera, used to load a preview of the location when the marker in the interactive map is clicked on.
  - The link is of the form `https://www.youtube.com/embed/ABCDEFGHIJK`, where `ABCDEFGHIJK` is the 11-character string used to identify the video ID on YouTube.
  - The recordings are 360-degree videos in spherical format corresponding to the files `S####_1min_Insta360.mp4` in the <a href="https://doi.org/10.21979/N9/AVHSBX">LCS dataset</a>.
- `vidLink_oneMin` : strings
  - The YouTube sharing link to the 1-minute excerpt of the full-length recording from the Insta360 One R camera, used to redirect the user to the same preview as `vidEmbed` hosted on YouTube.
  - The link is of the form `https://youtu.be/ABCDEFGHIJK`, where `ABCDEFGHIJK` is the 11-character string used to identify the video ID on YouTube, and should match the 11-character string for `vidEmbed`.
  - The recordings are 360-degree videos in spherical format corresponding to the files `S####_1min_Insta360.mp4` in the <a href="https://doi.org/10.21979/N9/AVHSBX">LCS dataset</a>.
- `vidLink_full` : strings
  - The YouTube sharing link to the full-length recording from the Insta360 One R camera, used to redirect the user to the full-length recording as hosted on YouTube.
  - The link is of the form `https://youtu.be/ABCDEFGHIJK`, where `ABCDEFGHIJK` is the 11-character string used to identify the video ID on YouTube.
  - The recordings are 360-degree videos in spherical format corresponding to the files `S####_full_Insta360.mp4` in the <a href="https://doi.org/10.21979/N9/AVHSBX">LCS dataset</a>.

## Admin page(s)

Currently, we have not set up an admin page for the application that the code in this repository builds, but the configuration in `.\visualisation\urls.py` is as follows:

  ```JavaScript
  019 urlpatterns = [
  020     path('admin/', admin.site.urls),
  021     path('', include('lioncitysoundscapes.urls'))
  022 ]
  ```
  
The `lioncitysoundscapes.urls` part on Line 21 indicates that `urls.py` will access `.\lioncitysoundscapes\urls.py` to load the related application URLs (following standard Python module naming conventions). Similarly, the `admin.site.urls` part on Line 20 indicates that `.\admin\site\urls.py` will be accessed to load the admin pages for the application if any (which do not exist yet, but you may add them in if you want).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- END OF DOCUMENT -->
