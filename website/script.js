document
  .getElementById("contentImageInput")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("contentImageDisplay").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("styleImageInput")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("styleImageDisplay").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("contentImageInput2")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("contentImageDisplay2").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("styleImageInput2")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("styleImageDisplay2").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("styleImageInput1")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("styleImageDisplay1").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });
