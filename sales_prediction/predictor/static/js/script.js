function updateProducts(selectedProduct = null) {
    const category = document.getElementById("category").value;
    const productSelect = document.getElementById("product");

    productSelect.innerHTML = "";

    if (!categoryProducts[category]) return;

    categoryProducts[category].forEach(product => {
        const option = document.createElement("option");
        option.value = product;
        option.text = product;

        // Restore selected product
        if (selectedProduct && product === selectedProduct) {
            option.selected = true;
        }

        productSelect.appendChild(option);
    });
}

document.addEventListener("DOMContentLoaded", function () {

    const selectedProduct = "{{ selected_product|default:'' }}";
    const selectedCategory = "{{ selected_category|default:'' }}";

    const categorySelect = document.getElementById("category");

    // Restore category first
    if (selectedCategory) {
        categorySelect.value = selectedCategory;
    }

    // Then update products and restore product
    updateProducts(selectedProduct);
});