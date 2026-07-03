# Documentation images

Drop arbitrary image assets here (PNG, JPG, SVG, GIF). Anything in this
directory is served by Sphinx via `html_static_path = ["_static"]` (see
`docs/conf.py`) and copied verbatim into `_build/html/_static/images/`.

## Referencing an image

From any `.rst` page, use an absolute (source-root-relative) path so the
reference works regardless of the page's depth in the tree:

```rst
.. figure:: /_static/images/example.png
   :alt: Short description of the figure
   :align: center
   :width: 80%

   Optional caption rendered below the image.
```

- `.. figure::` adds a caption + is referenceable; `.. image::` is bare.
- The leading `/` means "relative to the docs source root" (`docs/`), not the
  filesystem root. A page-relative path (`../_static/...`) also works but is
  fragile across the Diátaxis quadrants.
- Sphinx copies *referenced* images to `_build/html/_images/` automatically;
  the `_static` path additionally serves everything here unreferenced (useful
  for the theme logo, favicons, raw-HTML `<img>`, etc.).

## From a MyST notebook tutorial

```markdown
![alt text](/_static/images/example.png)
```
