Gallery
=======

Each scene is a learned waxMorph trajectory: a population of spheroidal cells
assembling a target morphology, rendered in your browser from the
reconstructed per-frame cell positions and colors. Click a scene to open the
player. Use the timeline to play, pause, or scrub, and drag to orbit.
``Camera: Fly`` switches to a free camera: click the view, then move with ``W``
``A`` ``S`` ``D`` and the mouse, ``E`` up, ``Q`` down, and ``Esc`` to release
the cursor.

.. raw:: html

   <script type="importmap">
   { "imports": { "three": "./_static/js/vendor/three.module.js" } }
   </script>
   <div id="wm-gallery" class="wm-gallery"></div>
   <noscript>The interactive gallery requires JavaScript and WebGL.</noscript>
   <script>
     /* ES modules and fetch are blocked on file://, so a page opened from disk
        would render a blank panel. Point the reader at a local HTTP server. */
     if (location.protocol === "file:") {
       document.getElementById("wm-gallery").innerHTML =
         '<div class="wm-gallery-notice"><p><b>Serve this page over HTTP to view the gallery.</b></p>' +
         '<p>Browsers block the 3D viewer\'s modules and data on <code>file://</code>. ' +
         'It works as published on the documentation site. To preview locally, run ' +
         '<code>python3 -m http.server</code> in this folder and open ' +
         '<code>http://localhost:8000/gallery.html</code>.</p></div>';
     }
   </script>
   <script type="module" src="_static/js/gallery-viewer.js"></script>
