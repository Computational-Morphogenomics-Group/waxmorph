Gallery
=======

Each scene below is a learned waxMorph trajectory — a population of spheroidal
cells assembling a target morphology — rendered live in your browser from the
reconstructed per-frame cell positions and colours. Click a scene to open the
player: use the timeline to play, pause, or scrub; **drag to orbit**, or switch
to the **free camera** (``Camera: Fly``) to move through the tissue with
``W`` ``A`` ``S`` ``D`` and the mouse (``Q``/``E`` for up and down, ``Esc`` to
release).

.. raw:: html

   <script type="importmap">
   { "imports": { "three": "./_static/js/vendor/three.module.js" } }
   </script>
   <div id="wm-gallery" class="wm-gallery"></div>
   <noscript>The interactive gallery requires JavaScript and WebGL.</noscript>
   <script>
     /* Opened from disk? ES modules + fetch are blocked on file:// — show how
        to preview locally instead of a blank panel. (No effect over http.) */
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
