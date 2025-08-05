# CalcVis Function Wizard
This is an interactive app powered by Streamlit that:
-takes a function input in standard typed form (i.e. 3x^2 + 5, or sin(6x), or ln(x), not every expression works)
-processes it and generates its differential and integral

**Live Demo** click here to go to open the app 

## Instructions to use Function Wizard:
1. **Enter a Function (using variable x)**
    -For example: x^2+3x-5; or sin(3x)
    -some notation will not work, though most should. 
        -Ln(x) just turns into Log(x)
        -Multiple exponents can cause issues (e^x^e^x), try being explicit (e^x)^(e^x)
        -Some notation/equations will generate differential and integral but will not graph
        -reverse bounds (b < a) will calculate properly, but the graph will not display accurately
2. **Adjust bounds** 
    -Enter values for a and b to set the bounds, should also adjust the visible scale of the x axis
    -You can also adjust the scale of the y axis
3. **Choose what to visualize** 
    -You can choose to graph the function, its differential, or both. 
    -By default the graph will show riemenn rectangles with n=50, you can adjust the slider for more or less rectangles. 
    -Select left, right, or midpoint for riemann rectangles
        -select from the dropdown menu
4. **To read Results**
    -Approximate area will be calculated based on the n-value (number of rectangles) you select
    -Exact area is calculated from the integral
    -The discrepancy between the two is labeled as "error"
    -If a > b the areas should be negative, althought the graph may not display it properly. 
5. **Repeat**
    -You can edit or replace the expression you want to visualize and it will(hopefully) update instantly
## Note: If you see an error code it means the program cannot parse or interpret the notation you used Try rewriting the expression in a different way. Being explicit with multiplication can help, although it should detect implicit multiplication between terms like 3x or (x+1)(x-1)



