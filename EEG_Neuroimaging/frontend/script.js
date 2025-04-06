async function upload() { 
  const file = document.getElementById("eegFile").files[0]; 
  const formData = new FormData(); 
  formData.append("file", file); 
 
  const res = await fetch("http://localhost:8000/predict", { 
    method: "POST", 
    body: formData 
  }); 
 
  const result = await res.json(); 
  document.getElementById("result").innerText = JSON.stringify(result, null, 2); 
} 
