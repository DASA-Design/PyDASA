Customization
========

This example demonstrates how to customize PyDASA by defining your own dimensional framework and tailoring the workflow to your specific needs.

Specifically we will create a custom dimensional framework called "CUSTOM" with the following fundamental dimensions:

    - **[T]**: Time (seconds).
    - **[D]**: Data Size (bits).
    - **[E]**: Effort (requests).

This FDUs are similar to the built-in "SOFTWARE" framework, but smaller and simpler for our demonstration purposes.

Then we will apply this custom framework to analyze a simple web server performance problem represented in a Queuing Model (M/M/c/K).