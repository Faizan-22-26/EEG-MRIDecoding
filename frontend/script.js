// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const eegFile = document.getElementById('eegFile');
  const eegName = document.getElementById('eegName');

  const patientName = document.getElementById('patientName');
  const patientAge = document.getElementById('patientAge');
  const patientGender = document.getElementById('patientGender');
  const diagnosis = document.getElementById('diagnosis');

  const sampleInfo = document.getElementById('sampleInfo');
  const samplePatient = document.getElementById('samplePatient');
  const sampleAge = document.getElementById('sampleAge');
  const sampleGender = document.getElementById('sampleGender');
  const sampleDiagnosis = document.getElementById('sampleDiagnosis');
  const sampleFileName = document.getElementById('sampleFileName');
  const sampleNameDisplay = document.getElementById('sampleNameDisplay');

  const eegTableBody = document.getElementById('eegTableBody');
  const processMRI = document.getElementById('processMRI');
  const mriSection = document.getElementById('mriSection');

  // MRI stats placeholders
  const resolutionVal = document.getElementById('resolutionVal');
  const timesliceVal = document.getElementById('timesliceVal');
  const sliceIndexVal = document.getElementById('sliceIndexVal');

  // Modal elements
  const sliceModal = document.getElementById('sliceModal');
  const modalImage = document.getElementById('modalImage');
  const modalSliceName = document.getElementById('modalSliceName');
  const closeModal = document.getElementById('closeModal');

  // Slices
  const axialSlice = document.getElementById('axialSlice');
  const sagittalSlice = document.getElementById('sagittalSlice');
  const coronalSlice = document.getElementById('coronalSlice');

  // Mock EEG channel data generator
  function generateEEGData() {
    const channels = ['Fz', 'Cz', 'Pz', 'Oz', 'T3', 'T4', 'F7', 'F8'];
    const bands = ['Delta', 'Theta', 'Alpha', 'Beta'];
    let rows = '';
    for (let i = 0; i < 8; i++) {
      const channel = channels[i] || `Ch${i+1}`;
      const band = bands[Math.floor(Math.random() * bands.length)];
      const peak = (Math.random() * 100).toFixed(2);
      const mean = (Math.random() * 50).toFixed(2);
      const std = (Math.random() * 10).toFixed(2);

      rows += `
        <tr>
          <td>${channel}</td>
          <td>${band}</td>
          <td>${peak}</td>
          <td>${mean}</td>
          <td>${std}</td>
        </tr>
      `;
    }
    return rows;
  }

  // 1. Upload EEG Sample (Simulated)
  uploadBtn.addEventListener('click', () => {
    const fileName = eegFile.value.split('\\').pop();
    const sampleTitle = eegName.value.trim();
    const pName = patientName.value.trim();
    const pAge = patientAge.value.trim();
    const pGender = patientGender.value;
    const pDiagnosis = diagnosis.value.trim();

    if (!fileName || !sampleTitle || !pName || !pAge || !pGender) {
      alert('Please fill in all required fields and select a file.');
      return;
    }

    // Simulate an upload
    alert(`EEG file "${fileName}" uploaded successfully as "${sampleTitle}".`);

    // Show sample info section
    sampleInfo.classList.remove('hidden');
    samplePatient.textContent = pName;
    sampleAge.textContent = pAge;
    sampleGender.textContent = pGender;
    sampleDiagnosis.textContent = pDiagnosis || 'Not specified';
    sampleFileName.textContent = fileName;
    sampleNameDisplay.textContent = sampleTitle;

    // Populate EEG table with mock data
    eegTableBody.innerHTML = generateEEGData();
  });

  // 2. Process MRI with simulated delay
  processMRI.addEventListener('click', () => {
    processMRI.innerText = 'Processing MRI...';
    processMRI.disabled = true;

    // Simulate delay (e.g., 2.5 seconds)
    setTimeout(() => {
      processMRI.innerText = 'Process MRI';
      processMRI.disabled = false;
      // Show the MRI section
      mriSection.classList.remove('hidden');

      // Set some mock stats
      resolutionVal.textContent = '256 x 256';
      timesliceVal.textContent = '42';
      sliceIndexVal.textContent = '5';
    }, 2500); // 2500ms delay
  });

  // 3. Clicking slices to expand (Modal)
  [axialSlice, sagittalSlice, coronalSlice].forEach(slice => {
    slice.addEventListener('click', () => {
      const imgEl = slice.querySelector('img');
      const sliceType = slice.querySelector('p').innerText;

      // Show modal
      sliceModal.classList.remove('hidden');
      modalImage.src = imgEl.src;
      modalSliceName.textContent = `Slice Type: ${sliceType}`;
    });
  });

  // 4. Close modal
  closeModal.addEventListener('click', () => {
    sliceModal.classList.add('hidden');
  });

  // Optionally close modal if user clicks outside modal content
  sliceModal.addEventListener('click', (e) => {
    if (e.target === sliceModal) {
      sliceModal.classList.add('hidden');
    }
  });
});
