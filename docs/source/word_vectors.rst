---
api
---

word\_vectors.read
------------------

.. automodule:: word_vectors.read
   :members:
   :undoc-members:
   :exclude-members: read, sniff, read_glove, read_w2v_text, read_w2v, read_dense

   .. autofunction:: read(f, file_type)
   .. autofunction:: read_glove(f)
   .. autofunction:: read_w2v_text(f)
   .. autofunction:: read_w2v(f)
   .. autofunction:: read_dense(f)
   .. autofunction:: sniff(f, buf_size)

word\_vectors.write
-------------------

.. automodule:: word_vectors.write
   :members:
   :undoc-members:
   :exclude-members: write, write_glove, write_w2v_text, write_w2v, write_dense

   .. autofunction:: write(wf, vocab, vectors, file_type, max_len)
   .. autofunction:: write_glove(wf, vocab, vectors)
   .. autofunction:: write_w2v_text(wf, vocab, vectors)
   .. autofunction:: write_w2v(wf, vocab, vectors)
   .. autofunction:: write_dense(wf, vocab, vectors, max_len)


word\_vectors.convert
---------------------

.. automodule:: word_vectors.convert
   :members:
   :undoc-members:

word\_vectors.utils
-------------------

.. automodule:: word_vectors.utils
   :members:
   :undoc-members:
   :exclude-members: is_binary

   .. autofunction:: is_binary(f, block_size, ratio)

Module contents
---------------

.. automodule:: word_vectors
   :members:
   :undoc-members:
